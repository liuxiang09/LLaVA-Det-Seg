from io import BytesIO
import torch
# import shutil # 移除文件操作库
# import random # 移除随机库
import matplotlib.pyplot as plt
import numpy as np
import cv2 # 或者使用 PIL
import argparse
import numpy as np
from torch import nn
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from LDS import LDSModel # 确保 LDSModel 类定义在 LDS.py 中
from PIL import Image
import requests
import sys
import re
import os
import json # 用于保存预测结果
from pycocotools.coco import COCO # 用于加载真实标注
from pycocotools.cocoeval import COCOeval # 用于评估
import pycocotools.mask as mask_util # 用于处理掩码

def parse_args(): # 修改参数解析，使其能从命令行接收 COCO 路径等
    parser = argparse.ArgumentParser(description="LDS Evaluation on COCO")
    parser.add_argument("--image_aspect_ratio", default="square")
    parser.add_argument("--coco_anno_path", type=str, required=True,
                        help="Path to COCO validation annotations file (e.g., instances_val2017.json)")
    parser.add_argument("--coco_image_dir", type=str, required=True,
                        help="Path to COCO validation images directory (e.g., val2017)")
    parser.add_argument("--output_results_path", type=str, default="lds_predictions.json",
                        help="Path to save the prediction results JSON file")
    # 您可能需要根据需要添加更多模型路径参数
    return parser.parse_args()

def load_clip_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        # 确保文件存在，并且是图像文件
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"Image file not found: {image_file}")
        try:
            image = Image.open(image_file).convert("RGB")
        except Exception as e:
            raise IOError(f"Error opening image file {image_file}: {e}")
    return image

# 简单的文本短语到 COCO 类别 ID 映射示例
# **注意：** 这是一个简化的示例，您需要根据您的模型输出和 COCO 类别进行更完善的映射
# 您可以预先运行模型在一些图片上看它可能输出哪些短语，然后手动建立映射
# 或者使用更高级的方法（如文本相似度）进行动态映射
# 如果无法映射，该预测将不会被包含在评估中（或被视为一个单独的未知类别）
phrase_to_coco_cat_id_map = {
    "person": 1,
    "bicycle": 2,
    "car": 3,
    "motorcycle": 4,
    "bus": 6,
    "truck": 8,
    "cat": 17,
    "dog": 18,
    "tv": 65, # Example mapping
    "laptop": 63,
    "book": 84,
    "chair": 62,
    "couch": 63, # Example, couch maps to same as laptop in some COCO versions, check yours!
    "bed": 65, # Example, bed maps to same as tv in some COCO versions
    "table": 67,
    "refrigerator": 72,
    "oven": 73,
    "sink": 75,
    "toilet": 77,
    "baseball bat": 39,
    "basketball": 37, # Example, find the correct ID for basketball if it's a separate category
}

# 获取 COCO 类别 ID 到名称的映射（用于调试或显示结果）
coco_cat_id_to_name_map = {
    cat['id']: cat['name'] for cat in COCO('/mnt/2753047e-bb0d-4a84-9488-1fce437519b3/coco/annotations/instances_val2017.json').loadCats()
}


def main():
    disable_torch_init()
    args = parse_args()

    # 检查文件和目录是否存在
    if not os.path.exists(args.coco_anno_path):
        print(f"Error: COCO annotations file not found at {args.coco_anno_path}")
        sys.exit(1)
    if not os.path.isdir(args.coco_image_dir):
        print(f"Error: COCO image directory not found at {args.coco_image_dir}")
        sys.exit(1)

    # 模型路径和名称
    llava_model_path = "checkpoints/llava-LDS-v3" # 请确保路径正确
    groundingdino_config_path = "groundingdino/config/GroundingDINO_SwinT_OGC.py" # 请确保路径正确
    groundingdino_checkpoint_path = "checkpoints/groundingdino_swint_ogc.pth" # 请确保路径正确
    # groundingdino_checkpoint_path = "/home/hpc/Desktop/GroundingDino-Finetuning/fine_tuning_output/checkpoint0000.pth" # 请确保路径正确
    sam_model_type = "vit_h"
    sam_checkpoint_path = "checkpoints/sam_vit_h_4b8939.pth" # 请确保路径正确

    # 实例化模型
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LDSModel(
        llava_model_path=llava_model_path,
        groundingdino_config_path=groundingdino_config_path,
        groundingdino_checkpoint_path=groundingdino_checkpoint_path,
        sam_model_type=sam_model_type,
        sam_checkpoint_path=sam_checkpoint_path,
        device=device,
        **{} # 移除 kwargs
    ).to(device)
    print("Model loaded.")

    # 加载 COCO 真实标注
    print(f"Loading COCO annotations from {args.coco_anno_path}...")
    coco_gt = COCO(args.coco_anno_path)
    print("COCO annotations loaded.")

    # 获取所有图像 ID
    img_ids = coco_gt.getImgIds()
    # img_ids = img_ids[:10] # 可选：先在少量图片上测试

    # 存储预测结果的列表，用于 COCO 评估
    coco_predictions = []

    print(f"Starting inference on {len(img_ids)} images...")
    for i, img_id in enumerate(img_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        image_path = os.path.join(args.coco_image_dir, img_info['file_name'])

        # 跳过无法加载的图像
        try:
            image_clip = load_clip_image(image_path)
            from groundingdino.util.inference import load_image, annotate
            image_source, image_det = load_image(image_path) # 确保 load_image 返回正确格式
        except (FileNotFoundError, IOError) as e:
            print(f"Skipping image {img_info['file_name']} due to error: {e}")
            continue

        # 构建 LLaVA 输入 prompt
        # 对于数据集评估，您可能需要一个策略来生成问题。
        # 简单起见，我们可以使用一个通用问题，或者尝试检测所有对象。
        # 您的模型通过解析 LLaVA 输出的 [DET] 指令来决定检测什么。
        # 我们可以给一个通用 prompt 促使它检测常见对象。
        # 例如："Please detect all objects in the image. [DET]"
        # 或者 "Describe the image and detect the main objects. [DET]"
        # 这里我们沿用您原有的 prompt 生成逻辑，但使用一个示例通用问题
        question = "Please identify all the objects in the image." # 示例问题
        # question = "person"
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv_mode = "llava_v1" # 确保与您的模型训练一致
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        llava_prompt_text = conv.get_prompt()


        images_tensor = process_images(
                image_clip,
                model.image_processor,
                model.kwargs # model.kwargs 可能需要根据您的模型定义进行调整
            ).to(model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(llava_prompt_text, model.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(model.device)
        )

        # 运行模型推理
        with torch.inference_mode():
            llava_output, groundingdino_output, sam_output = model.eval_forward(
                images_tensor, input_ids, image_source, image_det
            )
        # 处理输出并收集预测结果
        if groundingdino_output != {} and sam_output != {}:
            boxes = groundingdino_output["boxes"] # 这些是归一化坐标 [cx, cy, w, h]
            phrases = groundingdino_output["phrases"] # 预测的文本短语列表
            masks = sam_output["masks"] # SAM 输出的 mask [N, 1, H, W]
            mask_scores = sam_output["mask_scores"] # SAM 输出的 mask 分数 [N, 1]

            # 确保 GroundingDINO 输出的框数量与 SAM 输出的掩码数量匹配
            if boxes.shape[0] != masks.shape[0]:
                 print(f"Warning: Image {img_id} - Mismatch between GDINO boxes ({boxes.shape[0]}) and SAM masks ({masks.shape[0]}). Skipping.")
                 continue

            for j in range(masks.shape[0]):
                predicted_mask = masks[j].squeeze().cpu().numpy() # 获取布尔掩码 (H, W)
                predicted_score = mask_scores[j].item() # 获取得分 (float)
                predicted_phrase = phrases[j] # 获取预测短语 (str)

                # 将预测短语映射到 COCO 类别 ID
                category_id = phrase_to_coco_cat_id_map.get(predicted_phrase.lower(), None) # 使用 .get 防止 key 不存在错误
                # category_id = 1
                # 如果无法映射到已知 COCO 类别，则跳过此预测
                if category_id is None:
                    # print(f"Warning: Phrase '{predicted_phrase}' not found in COCO category map. Skipping prediction.")
                    continue # 跳过无法映射的预测

                # 将预测掩码转换为 COCO RLE 格式
                # 需要将 numpy bool 掩码转换为 uint8 Fortran contiguous 格式
                rle_mask = mask_util.encode(np.asfortranarray(predicted_mask.astype(np.uint8)))

                # 计算预测掩码的边界框 (可选，COCO 评估主要使用掩码，但结果格式需要 bbox)
                bbox = mask_util.toBbox(rle_mask).tolist() # [x, y, w, h]

                # 将预测结果添加到列表中
                coco_predictions.append({
                    "image_id": img_id,
                    "category_id": category_id,
                    "segmentation": rle_mask,
                    "score": predicted_score,
                    "bbox": bbox # 可选，根据实际评估工具需求添加
                })

        import pdb 
        # pdb.set_trace()
        if (i + 1) % 10 == 0: # 每处理一定数量的图像打印进度
            print(f"Processed {i + 1}/{len(img_ids)} images.")

    print("Inference finished.")
    print(f"Collected {len(coco_predictions)} predictions.")

    # 保存预测结果到 JSON 文件 (COCO 结果格式)
    print(f"Saving predictions to {args.output_results_path}...")
    # COCO API 需要 RLE 中的 bytes 类型，但 JSON 不能直接序列化 bytes。
    # 通常，您需要将 RLE 转换为可以序列化的格式（例如，将其 'counts' 字段解码为字符串）。
    # 或者使用 pycocotools 提供的 dumping 函数，如果它支持的话。
    # 这里我们手动处理一下，将 bytes RLE 转换为字符串
    for pred in coco_predictions:
         pred['segmentation']['counts'] = pred['segmentation']['counts'].decode('utf-8')

    with open(args.output_results_path, "w") as f:
        json.dump(coco_predictions, f)
    print("Predictions saved.")

    # 使用 COCO API 进行评估
    print("Starting COCO evaluation...")
    coco_dt = coco_gt.loadRes(args.output_results_path) # 加载预测结果

    # 后面的 COCO 评估代码保持不变
    # 使用 COCO API 进行评估
    print("Starting COCO evaluation...")
    # ... (原来的评估代码)
    # 创建 COCOeval 对象
    # iotDets 为 True 表示评估实例分割 (masks)
    coco_eval = COCOeval(coco_gt, coco_dt, 'segm') # 'segm' 表示分割评估

    # 如果您只评估部分类别，可以设置 params.catIds
    # 如果您使用了上面的 category_id 映射，COCOeval 会自动只评估这些类别
    coco_eval.params.catIds = list(phrase_to_coco_cat_id_map.values()) # 仅评估有映射的类别
    # 运行评估
    coco_eval.evaluate()
    coco_eval.accumulate()
    print("COCO evaluation results:")
    coco_eval.summarize()

    # 格式化输出结果 (summarize 已经提供了标准格式)
    # 您可以进一步解析 coco_eval.stats 数组来获取更详细的指标
    # coco_eval.stats 包含了 [AP @[ IoU=0.50:0.95 | area= all | maxDets=100 ]
    #                  [AP @[ IoU=0.50      | area= all | maxDets=100 ]
    #                  [AP @[ IoU=0.75      | area= all | maxDets=100 ]
    #                  [AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ]
    #                  [AP @[ IoU=0.50:0.95 | area= medium | maxDets=100 ]
    #                  [AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ]
    #                  [AR @[ IoU=0.50:0.95 | area= all | maxDets=  1 ]
    #                  [AR @[ IoU=0.50:0.95 | area= all | maxDets= 10 ]
    #                  [AR @[ IoU=0.50:0.95 | area= all | maxDets=100 ]
    #                  [AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ]
    #                  [AR @[ IoU=0.50:0.95 | area= medium | maxDets=100 ]
    #                  [AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ]]

    # print("\n--- Formatted mAP Results ---")
    # print(f"Average Precision  (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = {coco_eval.stats[0]:.3f}")
    # print(f"Average Precision  (AP) @[ IoU=0.50      | area= all | maxDets=100 ] = {coco_eval.stats[1]:.3f}")
    # print(f"Average Precision  (AP) @[ IoU=0.75      | area= all | maxDets=100 ] = {coco_eval.stats[2]:.3f}")
    # print(f"Average Recall     (AR) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = {coco_eval.stats[8]:.3f}") # AR@100 maxDets


if __name__ == "__main__":
    # 调用时需要在命令行传入 COCO 标注文件和图片文件夹路径
    # 例如：python your_script_name.py --coco_anno_path /path/to/coco/annotations/instances_val2017.json --coco_image_dir /path/to/coco/val2017
    main()