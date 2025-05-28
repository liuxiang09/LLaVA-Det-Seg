import cv2
import numpy as np
from typing import List, Tuple, Union
import uuid

def parse_detection_result(text: str) -> List[Tuple[str, List[float]]]:



    """解析检测结果文本，返回标签和边界框列表"""
    results = []
    lines = text.strip().split('\n')
    for line in lines:
        if '<box>' not in line or '</box>' not in line:
            continue
        # 清理标签
        label = line.split(':')[0].strip('- ').strip()
        if not label:
            continue
            
        # 提取并解析边界框坐标
        box_str = line.split('<box>')[1].split('</box>')[0]
        coords = box_str.strip('[]').split(',')
        if len(coords) != 4:  # 确保有4个坐标值
            continue
            
        # 转换坐标为浮点数
        box = []
        for x in coords:
            x = x.strip()
            if x:  # 只处理非空字符串
                box.append(float(x))
                
        if len(box) == 4:  # 只添加完整的检测框
            results.append((label, box))
                
    # print("text:", text)
    # print("results:", results)
    return results

def draw_coco_boxes(image_path: str, 
                   boxes: Union[List[List[float]], List[Tuple[str, List[float]]]], 
                   labels: List[str] = None):
    """
    在图片上绘制 COCO 格式的边界框和标签
    Args:
        image_path: 图片路径
        boxes: 边界框列表，可以是 [x,y,w,h] 列表或 (label,[x,y,w,h]) 元组列表
        labels: 标签列表（当 boxes 不包含标签时使用）
    """

    # 生成唯一文件名
    # unique_id = str(uuid.uuid4())[:8]
    # output_path = f"result_{unique_id}.jpg"
    output_path = "result.jpg"
    # 读取图片
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    # 设置字体和文本参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_color = (255, 255, 255)  # 白色文本
    
    # 处理输入数据格式
    if isinstance(boxes[0], tuple):  # (label, box) 格式
        labeled_boxes = boxes
    else:  # 分离的 boxes 和 labels 格式
        labeled_boxes = zip(labels if labels else [''] * len(boxes), boxes)
    
    # 绘制边界框和标签
    for label, box in labeled_boxes:
        x_center, y_center, w, h = box
        
        # 转换为左上角和右下角坐标
        x1 = int((x_center - w/2) * width)
        y1 = int((y_center - h/2) * height)
        x2 = int((x_center + w/2) * width)
        y2 = int((y_center + h/2) * height)
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 准备标签文本
        if label:
            # 获取文本大小
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness)
            
            # 绘制标签背景
            cv2.rectangle(image, 
                         (x1, y1 - text_height - 5), 
                         (x1 + text_width, y1), 
                         (0, 255, 0), 
                         -1)  # -1 表示填充矩形
            
            # 绘制标签文本
            cv2.putText(image, 
                       label, 
                       (x1, y1 - 5), 
                       font, 
                       font_scale, 
                       text_color, 
                       font_thickness)
    
    # 保存结果
    cv2.imwrite(output_path, image)
    return output_path

def paint(detection_boxes: str, image_path: str):
    # 示例检测结果文本
    # detection_text = """
    # - person: <box>[0.499, 0.501, 0.998, 0.998]</box>
    # """

    # 解析检测结果并绘制
    boxes_with_labels = parse_detection_result(detection_boxes)
    return draw_coco_boxes(image_path, boxes_with_labels)
    
if __name__ == "__main__":
    # 示例检测结果文本
    detection_text = """
    - car: <box>[0.712, 0.586, 0.576, 0.578]</box> 
    - car: <box>[0.158, 0.538, 0.316, 0.42]</box> 
    - person: <box>[0.545, 0.405, 0.223, 0.621]</box>
    """
    
    # 图片路径
    image_path = "/home/hpc/Desktop/LLaVA/llava/serve/examples/extreme_ironing.jpg"
    
    # 调用绘制函数
    paint(detection_text, image_path)