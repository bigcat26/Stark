"""
使用 ONNX Runtime 进行模板匹配
"""
import cv2
import numpy as np
import argparse
import onnxruntime as ort
from typing import Tuple


def preprocess_image(img_bgr: np.ndarray, target_sz: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    预处理图片：Resize -> Normalize -> ToTensor
    
    Args:
        img_bgr: BGR 格式的图片 (H, W, 3)
        target_sz: 目标尺寸 (正方形)
    
    Returns:
        img_tensor: 预处理后的图片 [1, 3, H, W]
        mask: mask [1, H, W] (全为 False，表示所有像素都有效)
    """
    # 1. Resize 到固定大小
    img_resized = cv2.resize(img_bgr, (target_sz, target_sz))
    
    # 2. BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # 3. 归一化到 [0, 1]
    img_norm = img_rgb.astype(np.float32) / 255.0
    
    # 4. ImageNet 标准化
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img_norm = (img_norm - mean) / std
    
    # 5. 转换为 [1, 3, H, W] 格式
    img_tensor = img_norm.transpose(2, 0, 1)[np.newaxis, :, :, :]  # [1, 3, H, W]
    
    # 6. 创建 mask (全为 False，表示所有像素都有效)
    mask = np.zeros((1, target_sz, target_sz), dtype=np.bool_)
    
    return img_tensor.astype(np.float32), mask


def run_match_onnx(template_path: str, search_path: str, onnx_file: str, 
                   output_path: str = "stark_result_onnx.jpg"):
    """
    使用 ONNX Runtime 进行模板匹配
    
    Args:
        template_path: 模板图片路径
        search_path: 搜索图片路径
        onnx_file: ONNX 模型文件路径
        output_path: 输出结果图片路径
    """
    print("=" * 50)
    print("STARK-S ONNX Runtime Inference")
    print("=" * 50)
    
    # 1. 加载 ONNX 模型
    print(f"Loading ONNX model from {onnx_file}...")
    try:
        # 尝试使用 GPU (如果可用)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(onnx_file, providers=providers)
        print(f"✓ Model loaded. Using: {session.get_providers()}")
    except Exception as e:
        print(f"⚠ GPU not available, using CPU: {e}")
        session = ort.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
    
    # 获取输入输出名称
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    print(f"Input names: {input_names}")
    print(f"Output names: {output_names}")
    
    # 2. 读取图片
    print(f"Loading images...")
    img_template = cv2.imread(template_path)
    img_search = cv2.imread(search_path)
    
    if img_template is None or img_search is None:
        print("Error: Failed to load images")
        return
    
    H_search, W_search = img_search.shape[:2]
    print(f"Template size: {img_template.shape[:2]}")
    print(f"Search size: {img_search.shape[:2]}")
    
    # 3. 预处理
    print("Preprocessing images...")
    # Template: 128x128
    img_z, mask_z = preprocess_image(img_template, 128)
    # Search: 320x320
    img_x, mask_x = preprocess_image(img_search, 320)
    
    print(f"Template tensor shape: {img_z.shape}, mask shape: {mask_z.shape}")
    print(f"Search tensor shape: {img_x.shape}, mask shape: {mask_x.shape}")
    
    # 4. 推理
    print("Running inference...")
    # ONNX 可能不支持 bool，使用 int64
    inputs = {
        'img_z': img_z,
        'mask_z': mask_z.astype(np.int64),
        'img_x': img_x,
        'mask_x': mask_x.astype(np.int64)
    }
    
    outputs = session.run(output_names, inputs)
    pred_boxes = outputs[0]  # [1, num_queries, 4]
    
    print(f"Prediction shape: {pred_boxes.shape}")
    
    # 5. 解析结果
    # pred_boxes 形状: [1, num_queries, 4]，其中 4 是 (cx, cy, w, h) 在 [0,1] 范围内
    pred_boxes = pred_boxes.reshape(-1, 4)  # [num_queries, 4]
    
    # 取所有预测框的平均值作为最终结果
    pred_box_norm = pred_boxes.mean(axis=0)  # [4] (cx, cy, w, h) 0~1
    
    print(f"Match completed (using mean of {pred_boxes.shape[0]} predictions)")
    print(f"Predicted box (normalized): cx={pred_box_norm[0]:.4f}, cy={pred_box_norm[1]:.4f}, "
          f"w={pred_box_norm[2]:.4f}, h={pred_box_norm[3]:.4f}")
    
    # 6. 还原坐标到原图
    cx, cy, w, h = pred_box_norm
    
    # 映射回原图尺寸
    real_cx = cx * W_search
    real_cy = cy * H_search
    real_w = w * W_search
    real_h = h * H_search
    
    x1 = int(real_cx - real_w / 2)
    y1 = int(real_cy - real_h / 2)
    x2 = int(real_cx + real_w / 2)
    y2 = int(real_cy + real_h / 2)
    
    # 确保坐标在图片范围内
    x1 = max(0, min(x1, W_search))
    y1 = max(0, min(y1, H_search))
    x2 = max(0, min(x2, W_search))
    y2 = max(0, min(y2, H_search))
    
    print(f"Predicted box (pixels): x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    
    # 7. 画图
    vis_img = img_search.copy()
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(vis_img, "Match", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # 8. 保存结果
    cv2.imwrite(output_path, vis_img)
    print(f"✓ Result saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Template matching using ONNX Runtime')
    parser.add_argument('--template', required=True, help='Path to template image')
    parser.add_argument('--search', required=True, help='Path to search image')
    parser.add_argument('--onnx', default='stark_s.onnx', help='Path to ONNX model file')
    parser.add_argument('--output', default='stark_result_onnx.jpg', 
                        help='Path to output result image')
    
    args = parser.parse_args()
    
    run_match_onnx(args.template, args.search, args.onnx, args.output)

