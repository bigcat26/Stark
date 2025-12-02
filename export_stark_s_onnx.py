"""
将 STARK-S 模型转换为 ONNX 格式
"""
import os
import sys
import torch
import argparse
import numpy as np
from lib.config.stark_s.config import cfg, update_config_from_file
from lib.models.stark.stark_s import build_starks
from lib.utils.misc import NestedTensor
from lib.utils.merge import merge_template_search

# 设置 checkpoint 加载环境
def setup_checkpoint_loading():
    """设置 checkpoint 加载环境，创建必要的占位符模块"""
    if 'lib.train.admin.local' not in sys.modules:
        local_file = os.path.join(os.getcwd(), 'lib', 'train', 'admin', 'local.py')
        if not os.path.exists(local_file):
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            with open(local_file, 'w') as f:
                f.write('class EnvironmentSettings:\n')
                f.write('    def __init__(self):\n')
                f.write('        pass\n')
        import importlib
        try:
            importlib.import_module('lib.train.admin.local')
        except ImportError:
            import types
            local_module = types.ModuleType('lib.train.admin.local')
            class EnvironmentSettings:
                def __init__(self):
                    pass
            local_module.EnvironmentSettings = EnvironmentSettings
            sys.modules['lib.train.admin.local'] = local_module

setup_checkpoint_loading()


class STARK_S_ONNX(torch.nn.Module):
    """
    用于 ONNX 导出的 STARK-S 模型包装器
    将整个推理流程封装为一个模型
    """
    def __init__(self, model):
        super(STARK_S_ONNX, self).__init__()
        self.model = model
        
    def forward(self, img_z, mask_z, img_x, mask_x):
        """
        Args:
            img_z: template image [1, 3, 128, 128]
            mask_z: template mask [1, 128, 128] (bool or int64)
            img_x: search image [1, 3, 320, 320]
            mask_x: search mask [1, 320, 320] (bool or int64)
        Returns:
            pred_boxes: [1, num_queries, 4] (cx, cy, w, h) in [0,1]
        """
        # 转换 mask 为 bool 类型（如果输入是 int64）
        if mask_z.dtype != torch.bool:
            mask_z = mask_z.bool()
        if mask_x.dtype != torch.bool:
            mask_x = mask_x.bool()
        
        # 创建 NestedTensor
        z_nested = NestedTensor(img_z, mask_z)
        x_nested = NestedTensor(img_x, mask_x)
        
        # 1. 分别处理 template 和 search 图像
        z_dict = self.model.forward_backbone(z_nested)
        x_dict = self.model.forward_backbone(x_nested)
        
        # 2. 合并 template 和 search 特征
        feat_dict_list = [z_dict, x_dict]
        seq_dict = merge_template_search(feat_dict_list)
        
        # 3. 运行 transformer 获取预测结果
        out_dict, _, _ = self.model.forward_transformer(seq_dict=seq_dict, run_box_head=True)
        
        return out_dict['pred_boxes']


def export_onnx(config_file, checkpoint_file, output_file, device='cpu'):
    """
    导出 STARK-S 模型为 ONNX 格式
    
    Args:
        config_file: 配置文件路径
        checkpoint_file: checkpoint 文件路径
        output_file: 输出的 ONNX 文件路径
        device: 设备 ('cpu', 'cuda')
    """
    print("=" * 50)
    print("STARK-S ONNX Export")
    print("=" * 50)
    
    # 1. 加载配置
    print(f"Loading config from {config_file}...")
    update_config_from_file(config_file)
    
    # 2. 构建模型
    print("Building model...")
    model = build_starks(cfg)
    
    # 3. 加载权重
    print(f"Loading weights from {checkpoint_file}...")
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    
    if 'net' in checkpoint:
        model.load_state_dict(checkpoint['net'], strict=False)
    else:
        model.load_state_dict(checkpoint['net_model'], strict=False)
    
    model = model.to(device)
    model.eval()
    
    # 4. 创建 ONNX 包装器
    onnx_model = STARK_S_ONNX(model)
    onnx_model.eval()
    
    # 5. 准备示例输入
    print("Preparing example inputs...")
    img_z = torch.randn(1, 3, 128, 128).to(device)
    # ONNX 可能不支持 bool，使用 int64
    mask_z = torch.zeros(1, 128, 128, dtype=torch.int64).to(device)
    img_x = torch.randn(1, 3, 320, 320).to(device)
    mask_x = torch.zeros(1, 320, 320, dtype=torch.int64).to(device)
    
    # 6. 测试前向传播
    print("Testing forward pass...")
    with torch.no_grad():
        output = onnx_model(img_z, mask_z, img_x, mask_x)
        print(f"Output shape: {output.shape}")
    
    # 7. 导出 ONNX
    print(f"Exporting to ONNX: {output_file}...")
    torch.onnx.export(
        onnx_model,
        (img_z, mask_z, img_x, mask_x),
        output_file,
        input_names=['img_z', 'mask_z', 'img_x', 'mask_x'],
        output_names=['pred_boxes'],
        dynamic_axes={
            'img_z': {0: 'batch_size'},
            'mask_z': {0: 'batch_size'},
            'img_x': {0: 'batch_size'},
            'mask_x': {0: 'batch_size'},
            'pred_boxes': {0: 'batch_size'}
        },
        opset_version=11,
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"✓ ONNX model exported successfully to {output_file}")
    
    # 8. 验证 ONNX 模型
    try:
        import onnx
        onnx_model_check = onnx.load(output_file)
        onnx.checker.check_model(onnx_model_check)
        print("✓ ONNX model validation passed")
    except ImportError:
        print("⚠ onnx package not installed, skipping validation")
    except Exception as e:
        print(f"⚠ ONNX validation warning: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export STARK-S model to ONNX format')
    parser.add_argument('--config', default='experiments/stark_s/baseline.yaml', 
                        help='Config file path')
    parser.add_argument('--checkpoint', 
                        default='checkpoints/train/stark_s/baseline/stark_s50_baseline.pth.tar',
                        help='Checkpoint file path')
    parser.add_argument('--output', default='stark_s.onnx', 
                        help='Output ONNX file path')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                        help='Device to use for export')
    
    args = parser.parse_args()
    
    export_onnx(args.config, args.checkpoint, args.output, args.device)

