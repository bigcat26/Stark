import os
import sys
import cv2
import torch
import numpy as np
import argparse
import torch.nn.functional as F
from PIL import Image
import types
import importlib
import importlib.util
import pickle

# 将 lib 加入路径
sys.path.append(os.getcwd())

# 在导入其他模块之前，先创建占位符模块
def setup_checkpoint_loading():
    """
    设置 checkpoint 加载环境，创建必要的占位符模块
    用于解决加载 checkpoint 时可能遇到的缺失模块问题
    """
    # 创建临时的 local.py 文件（如果不存在）
    local_file = os.path.join(os.getcwd(), 'lib', 'train', 'admin', 'local.py')
    if not os.path.exists(local_file):
        # 确保目录存在
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        # 创建一个最小的 local.py 文件
        with open(local_file, 'w') as f:
            f.write('class EnvironmentSettings:\n')
            f.write('    def __init__(self):\n')
            f.write('        pass\n')
    
    # 尝试导入模块（如果还没有导入）
    if 'lib.train.admin.local' not in sys.modules:
        try:
            importlib.import_module('lib.train.admin.local')
        except ImportError:
            # 如果导入失败，创建一个占位符模块
            local_module = types.ModuleType('lib.train.admin.local')
            class EnvironmentSettings:
                def __init__(self):
                    pass
            local_module.EnvironmentSettings = EnvironmentSettings
            sys.modules['lib.train.admin.local'] = local_module

# 在导入其他模块之前就设置好占位符模块
setup_checkpoint_loading()

from lib.config.stark_s.config import cfg, update_config_from_file
from lib.models.stark.stark_s import build_starks
from lib.utils.box_ops import box_cxcywh_to_xyxy

def safe_load_checkpoint(checkpoint_file, map_location):
    """
    安全加载 checkpoint，处理缺失模块的问题
    """
    # 确保占位符模块已创建
    setup_checkpoint_loading()
    
    try:
        # 尝试正常加载
        return torch.load(checkpoint_file, map_location=map_location, weights_only=False)
    except (ModuleNotFoundError, AttributeError) as e:
        if 'lib.train.admin.local' in str(e):
            # 如果是因为 lib.train.admin.local 缺失，重新设置并重试
            print(f"Warning: {e}, retrying with module setup...")
            setup_checkpoint_loading()
            # 再次尝试加载
            return torch.load(checkpoint_file, map_location=map_location, weights_only=False)
        else:
            raise

def get_device(device_arg=None):
    """
    获取设备，优先级: cuda > mps > cpu
    如果指定了 device_arg，则使用指定的设备
    """
    if device_arg:
        if device_arg == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        elif device_arg == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        elif device_arg == 'cpu':
            return torch.device('cpu')
        else:
            print(f"Warning: Device '{device_arg}' not available, falling back to auto selection")
    
    # 自动选择设备，优先级: cuda > mps > cpu
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def preprocess_image(img_bgr, target_sz):
    """
    预处理图片：Resize -> Normalize -> ToTensor
    STARK 要求 ImageNet 的均值和方差
    """
    # Resize
    img_resized = cv2.resize(img_bgr, (target_sz, target_sz))
    
    # BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize (ImageNet stats)
    img_norm = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_norm - mean) / std
    
    # To Tensor [C, H, W]
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float()
    
    # Add Batch Dim [1, C, H, W]
    return img_tensor.unsqueeze(0)

def run_demo(template_path, search_path, config_file, checkpoint_file, device=None):
    # 1. 加载配置
    update_config_from_file(config_file)
    
    # 2. 获取设备
    if device is None:
        device = get_device()
    else:
        device = get_device(device)
    print(f"Using device: {device}")
    
    # 3. 构建模型
    # STARK-S (Spatial only) 是最适合单帧匹配的
    model = build_starks(cfg)
    
    # 4. 加载权重
    print(f"Loading weights from {checkpoint_file}...")
    checkpoint = safe_load_checkpoint(checkpoint_file, device)
    
    # 处理 key 名字不匹配的问题 (去除 'net.' 前缀)
    if 'net' in checkpoint:
        model.load_state_dict(checkpoint['net'], strict=False)
    else:
        model.load_state_dict(checkpoint['net_model'], strict=False)
        
    # 将模型移动到指定设备
    model = model.to(device)
    model.eval()

    # 4. 读取图片
    img_template = cv2.imread(template_path)
    img_search = cv2.imread(search_path)
    
    if img_template is None or img_search is None:
        print("Error reading images")
        return

    H_search, W_search = img_search.shape[:2]

    # 5. 预处理
    # STARK 默认输入: Template=128, Search=320
    # 注意：这里我们暴力 resize 了全图，这会导致长宽比失真。
    # 正式的工程做法应该 Pad 到正方形再 Resize，或者用滑窗。
    z_tensor = preprocess_image(img_template, 128).to(device)
    x_tensor = preprocess_image(img_search, 320).to(device)

    # 6. 推理
    # 构造 NestedTensor (STARK 需要 mask，这里全为0表示全图有效)
    from lib.utils.misc import NestedTensor
    from lib.utils.merge import merge_template_search
    
    # Mask: 0 表示有效像素，1 表示 Padding。这里全是有效。
    mask_z = torch.zeros((1, 128, 128), dtype=torch.bool).to(z_tensor.device)
    mask_x = torch.zeros((1, 320, 320), dtype=torch.bool).to(x_tensor.device)
    
    z_nested = NestedTensor(z_tensor, mask_z)
    x_nested = NestedTensor(x_tensor, mask_x)

    with torch.no_grad():
        # 1. 分别处理 template 和 search 图像
        z_dict = model.forward_backbone(z_nested)
        x_dict = model.forward_backbone(x_nested)
        
        # 2. 合并 template 和 search 特征
        feat_dict_list = [z_dict, x_dict]
        seq_dict = merge_template_search(feat_dict_list)
        
        # 3. 运行 transformer 获取预测结果
        # STARK-S 模型返回 (out_dict, outputs_coord, output_embed)
        # out_dict 包含 'pred_boxes'，但没有 'pred_logits'（只有 STARK-ST 才有）
        out_dict, outputs_coord, _ = model.forward_transformer(seq_dict=seq_dict, run_box_head=True)

    # 7. 解析结果
    # pred_boxes 形状: (batch, num_queries, 4)，其中 4 是 (cx, cy, w, h) 在 [0,1] 范围内
    pred_boxes = out_dict['pred_boxes']  # [1, num_queries, 4]
    pred_boxes = pred_boxes.view(-1, 4)  # [num_queries, 4]
    
    # STARK-S 没有 pred_logits，所以取所有预测框的平均值作为最终结果
    # 或者可以选择第一个预测框
    pred_box_norm = pred_boxes.mean(dim=0).cpu().numpy()  # [4] (cx, cy, w, h) 0~1
    
    print(f"Match completed (using mean of {pred_boxes.shape[0]} predictions)")

    # 8. 还原坐标到原图
    cx, cy, w, h = pred_box_norm
    
    # 映射回原图尺寸
    real_cx = cx * W_search
    real_cy = cy * H_search
    real_w  = w * W_search
    real_h  = h * H_search
    
    x1 = int(real_cx - real_w / 2)
    y1 = int(real_cy - real_h / 2)
    x2 = int(real_cx + real_w / 2)
    y2 = int(real_cy + real_h / 2)

    # 9. 画图
    vis_img = img_search.copy()
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(vis_img, "Match", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    output_path = "stark_result.jpg"
    cv2.imwrite(output_path, vis_img)
    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True, help='Path to template image')
    parser.add_argument('--search', required=True, help='Path to search image')
    # 默认使用 baseline 配置
    parser.add_argument('--config', default='experiments/stark_s/baseline.yaml', help='Config file')
    # 你的权重文件路径
    parser.add_argument('--checkpoint', default='checkpoints/train/stark_s/baseline/stark_s50_baseline.pth.tar', help='Path to .pth model')
    parser.add_argument('--device', default=None, choices=['cuda', 'mps', 'cpu'], 
                        help='Device to use (cuda/mps/cpu). If not specified, auto-selects with priority: cuda > mps > cpu')
    
    args = parser.parse_args()
    
    run_demo(args.template, args.search, args.config, args.checkpoint, args.device)
