# tools/lr_finder_hook.py (最终健壮、无冲突版)

import sys
import math
import matplotlib.pyplot as plt
from mmcv.runner import Hook, HOOKS

@HOOKS.register_module()
class CustomLrFinderHook(Hook):
    """
    一个用于寻找最佳学习率的自定义Hook (DDP安全且无冲突版)。
    它在内部维护学习率状态，以避免与其他Hook冲突。
    """
    def __init__(self, start_lr=1e-8, end_lr=1.0, num_iters=100, by_epoch=False):
        if by_epoch:
            raise ValueError("CustomLrFinderHook only supports 'by_iter' mode.")
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_iters = num_iters
        self.history = {"lr": [], "loss": []}
        self.best_loss = float('inf')

    def before_run(self, runner):
        runner.logger.info(f"Starting CUSTOM LR Finder for {self.num_iters} iterations...")
        # 保存初始状态
        self.model_state = runner.model.state_dict()
        self.optimizer_state = runner.optimizer.state_dict()
        
        # 【核心修改】: 在Hook内部维护当前学习率的状态
        self.current_lr = self.start_lr
        for param_group in runner.optimizer.param_groups:
            param_group['lr'] = self.current_lr
        
        self.lr_multiplier = (self.end_lr / self.start_lr) ** (1.0 / self.num_iters)

    def after_train_iter(self, runner):
        if hasattr(runner, 'should_stop') and runner.should_stop:
            return

        loss = runner.outputs['loss'].item()
        self.history['lr'].append(self.current_lr) # 记录我们自己维护的lr
        self.history['loss'].append(loss)
        
        # 更新最佳损失
        if loss < self.best_loss and len(self.history['loss']) > 10:
            self.best_loss = loss
        
        # 检查损失是否发散
        if len(self.history['loss']) > 10 and loss > 4 * self.best_loss:
            runner.logger.warning("Loss is diverging. Stopping LR Finder.")
            self._plot_and_stop(runner)
            return

        # 【核心修改】: 更新我们自己维护的lr，并强制应用到优化器
        self.current_lr *= self.lr_multiplier
        for param_group in runner.optimizer.param_groups:
            param_group['lr'] = self.current_lr

        # 检查是否达到迭代次数
        if runner.iter + 1 >= self.num_iters:
            runner.logger.info("LR Finder finished.")
            self._plot_and_stop(runner)

    def _plot_and_stop(self, runner):
        if runner.rank == 0:
            runner.logger.info("Plotting LR-Loss curve...")
            lrs = self.history['lr']
            losses = self.history['loss']
            plt.figure()
            plt.plot(lrs[10:-5], losses[10:-5])
            plt.xscale('log')
            plt.xlabel('Learning Rate')
            plt.ylabel('Loss')
            plt.grid(True)
            save_path = f"{runner.work_dir}/custom_lr_find.png"
            plt.savefig(save_path)
            runner.logger.info(f"LR Finder plot saved to {save_path}")
            runner.logger.info("Please check the plot and update your main config file with the best LR.")
        
        runner.should_stop = True