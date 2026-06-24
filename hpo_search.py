import optuna
import os
import os.path as osp
from mmcv import Config
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.apis import train_detector
from mmcv.runner import set_random_seed, get_dist_info, Hook
from mmcv.runner import HOOKS # 导入 HOOKS 注册器
import torch
import warnings
import traceback # 导入 traceback 以打印详细错误

warnings.filterwarnings("ignore", "train_cfg and test_cfg is deprecated")
warnings.filterwarnings("ignore", "The `iou_calculator` is deprecated")

# ==============================================================================
# 1. 定义一个自定义 Hook 来检测 NaN Loss
# ==============================================================================
@HOOKS.register_module() # 将这个类注册到 MMCV 的 HOOKS 中，以便通过字符串'DetectNanHook'来调用
class DetectNanHook(Hook):
    """
    一个在每次训练迭代后检查损失是否为 NaN 的 Hook。
    如果损失为 NaN，它会抛出一个异常来中止训练。
    """
    def after_train_iter(self, runner):
        # runner.outputs 包含了当前迭代的输出，如 loss, log_vars 等
        loss_val = runner.outputs.get('loss')
        
        # 检查 loss 是否存在且为 NaN
        if loss_val is not None and torch.isnan(loss_val):
            # 打印信息并抛出异常
            print(f"\n\n!!! Trial #{runner.meta['trial_number']} failed: Loss became NaN at iteration {runner.iter}. Stopping this trial. !!!\n\n")
            # 抛出 ValueError，这个异常会被 objective 函数中的 try-except 块捕获
            raise ValueError(f"Loss became NaN at iteration {runner.iter}")

# ==============================================================================

def objective(trial: optuna.trial.Trial):
    # 1. 定义超参数的搜索空间
    depths_str = trial.suggest_categorical("depths", [
        "[2,2,2,5]"
    ])
    depths = eval(depths_str)

    # =========== 修改位置开始 ===========
    # 将学习率固定为单一值 0.00109
    # 使用 suggest_categorical 传入列表，既固定了值，又能被 Optuna 记录到日志中
    lr = trial.suggest_categorical("lr", [0.00008])
    # =========== 修改位置结束 ===========

    # 2. 加载并动态修改配置
    cfg = Config.fromfile('/home/lq/MM_Comparison/mmrotate-main/configs/ARConv-FEFM/DCNV4_Mamba_advance_ORCNN_CUDA_syBN.py')
    
    cfg.model.backbone.depths = depths
    cfg.optimizer.lr = lr
    
    # 更新工作目录命名，加入参数信息
    trial_work_dir = osp.join(cfg.work_dir, f'trial_{trial.number}_depths_{depths_str}_lr_{lr:.6f}')
    os.makedirs(trial_work_dir, exist_ok=True)
    cfg.work_dir = trial_work_dir

    # ==============================================================================
    # 2.1. ✅ 在配置中注册我们的自定义 Hook
    # ==============================================================================
    # 如果配置文件中没有 custom_hooks，就初始化一个空列表
    if 'custom_hooks' not in cfg:
        cfg.custom_hooks = []
    # 添加我们的 NaN 检测 Hook
    cfg.custom_hooks.append(dict(type='DetectNanHook', priority='VERY_LOW'))
    # ==============================================================================

    print(f"\n===== Starting Trial #{trial.number} ===== Params: depths={depths}, lr={lr:.6f} Work Dir: {cfg.work_dir} =================================\n")

    try:
        # 3. 开始训练和评估
        set_random_seed(0, deterministic=False)
        model = build_detector(cfg.model)
        
        if hasattr(model, 'init_weights'):
            model.init_weights()

        datasets = [build_dataset(cfg.data.train)]
        
        # 将 trial number 传入 meta，以便 Hook 内部可以访问
        meta = {'trial_number': trial.number}
        
        train_detector(model, datasets, cfg, distributed=False, validate=True, meta=meta)
        
        # 4. 获取并返回评估结果 (mAP)
        latest_map = model.runner.log_buffer.get_val('mAP')
        if latest_map is None: return 0.0
        return latest_map

    except Exception as e:
        # 当试验失败时（包括我们主动抛出的 NaN 异常），打印出详细的错误追溯信息
        print(f"Trial #{trial.number} failed with a DETAILED exception:")
        print(traceback.format_exc())
        # 返回一个差的值，让 Optuna 知道这个试验不成功
        return 0.0

if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    
    # 设置试验次数
    n_trials = 1
    study.optimize(objective, n_trials=n_trials)

    print("\n\n==================================================")
    print("Hyperparameter optimization finished!")
    print(f"Number of finished trials: {len(study.trials)}")