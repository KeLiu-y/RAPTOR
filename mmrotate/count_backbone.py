import torch
from mmcv import Config
from mmrotate.models import build_detector

# 尝试适配不同版本的 mmcv 导入路径
try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    try:
        from mmcv.cnn.utils import get_model_complexity_info
    except ImportError:
        print("Error: 无法找到 get_model_complexity_info，请检查 mmcv 版本")

def count_backbone_complexity(config_path, input_shape=(3, 1024, 1024)):
    # 1. 读取配置文件 (旧版使用 mmcv.Config)
    print(f"正在读取配置: {config_path}")
    cfg = Config.fromfile(config_path)
    
    # 2. 构建模型 (旧版使用 build_detector)
    model = build_detector(cfg.model)
    
    # 3. 提取 Backbone
    if hasattr(model, 'backbone'):
        backbone = model.backbone
    else:
        raise AttributeError("该模型没有名为 'backbone' 的属性，请检查模型结构。")

    if torch.cuda.is_available():
        backbone = backbone.cuda()

    print(f"正在分析 Backbone: {type(backbone).__name__} ...")
    print(f"输入尺寸: {input_shape}")

    # 4. 计算复杂度 (旧版 API 参数略有不同)
    # mmcv 1.x 的 get_model_complexity_info 返回 (flops_str, params_str)
    flops, params = get_model_complexity_info(
        backbone, 
        input_shape, 
        as_strings=True, 
        print_per_layer_stat=True  # 设置为 True 会打印每一层的详细参数
    )

    print("\n" + "="*40)
    print(f"Backbone FLOPs: {flops}")
    print(f"Backbone Params: {params}")
    print("="*40 + "\n")
if __name__ == '__main__':
    # --- 在这里修改你的配置路径 ---
    config_file = '/home/lq/MM_Comparison/mmrotate-main/configs/ARConv-FEFM/DCNv4_ad_ORCNN_small.py'
    
    input_shape = (3, 1024, 1024)
    
    try:
        count_backbone_complexity(config_file, input_shape)
    except FileNotFoundError:
        print(f"错误: 找不到配置文件 '{config_file}'，请在代码最后几行修改 config_file 变量。")