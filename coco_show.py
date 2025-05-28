import os
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
# 如果你之前安装了opencv，也可以用cv2
# import cv2

def visualize_coco_gt_masks(annotation_path: str, image_dir: str, image_id: int, desired_category_ids: list = None):
    """
    加载 COCO 地面真相标注并可视化特定图片的分割掩码。

    Args:
        annotation_path (str): COCO 地面真相标注文件的路径
                                (例如 'path/to/instances_val2017.json').
        image_dir (str): COCO 图片文件所在的目录路径
                         (例如 'path/to/val2017').
        image_id (int): 要可视化的图片的 COCO image_id.
        desired_category_ids (list, optional): 要可视化的特定类别ID列表。
                                                如果为 None，则可视化所有类别。默认为 None。
    """
    # 加载 COCO 标注
    coco = COCO(annotation_path)

    # 获取图片信息
    img_info = coco.loadImgs(image_id)[0]
    image_path = os.path.join(image_dir, img_info['file_name'])

    # 加载图片
    try:
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return

    # 获取该图片的所有标注
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)

    # 创建用于绘制掩码的叠加层
    mask_overlay = np.zeros_like(image, dtype=np.uint8)
    alpha = 0.5 # 叠加透明度

    print(f"Visualizing GT masks for image {image_id} (File: {img_info['file_name']}).")
    if desired_category_ids is None:
        print(f"Visualizing all {len(anns)} annotations.")
    else:
        # 获取类别名称以便打印
        cat_names = [coco.loadCats(cat_id)[0]['name'] for cat_id in desired_category_ids]
        print(f"Visualizing annotations for categories: {cat_names} (IDs: {desired_category_ids}).")


    # 绘制每个标注的掩码
    drawn_count = 0
    for ann in anns:
        category_id = ann['category_id']

        # === 添加的条件判断 ===
        if desired_category_ids is not None and category_id not in desired_category_ids:
            # 如果指定了类别ID列表，且当前标注的类别不在列表中，则跳过
            continue
        # ======================

        # annToMask 可以处理 RLE 和多边形格式的分割标注
        mask = coco.annToMask(ann)

        # 获取类别名称（可选，用于打印信息）
        category_name = coco.loadCats(category_id)[0]['name']

        # 为每个实例生成随机颜色
        color = [random.randint(0, 255) for _ in range(3)]

        # 将颜色应用到掩码区域
        for c in range(3):
             mask_overlay[:,:,c] = np.where(mask == 1, color[c], mask_overlay[:,:,c])

        # 可以在这里打印每个绘制的标注信息
        print(f" - Drawn GT object: category_id={category_id} ({category_name}), ann_id={ann['id']}")
        drawn_count += 1

    print(f"Finished drawing {drawn_count} masks.")

    # 将原始图片和掩码叠加层混合
    blended_image = image * (1 - alpha) + mask_overlay * alpha
    blended_image = blended_image.astype(np.uint8)

    # 显示结果
    plt.figure(figsize=(12, 8))
    plt.imshow(blended_image)
    plt.title(f"COCO GT Masks for Image ID: {image_id}")
    plt.axis('off')
    plt.show()

# --- 使用示例 ---
# 替换为你的文件路径
coco_annotation_path = '/mnt/2753047e-bb0d-4a84-9488-1fce437519b3/coco/annotations/instances_val2017.json'
coco_image_directory = '/mnt/2753047e-bb0d-4a84-9488-1fce437519b3/coco/val2017'
target_image_id = 397133 # 你想查看的 image_id

# 示例 1: 可视化所有类别 (默认行为)
# visualize_coco_gt_masks(coco_annotation_path, coco_image_directory, target_image_id)

# 示例 2: 只可视化类别 ID 为 1 (person) 的掩码
# 你可以在这里添加更多你想可视化的类别ID，比如 [1, 73]
visualize_coco_gt_masks(coco_annotation_path, coco_image_directory, target_image_id, desired_category_ids=[1])


