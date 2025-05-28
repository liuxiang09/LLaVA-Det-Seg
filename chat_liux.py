from io import BytesIO
import torch
import cv2
import shutil
import random
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
from LDS import LDSModel
from PIL import Image
import requests
import sys
import re
import os

def parse_args(args):
    parser = argparse.ArgumentParser(description="LDS chat")
    parser.add_argument("--image_aspect_ratio", default="square")
    return parser.parse_args(args)

def load_clip_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def main(args):
    disable_torch_init()


    args = parse_args(args)
    kwargs = {}
    # 模型路径和名称
    llava_model_path = "checkpoints/llava-LDS-v3"
    groundingdino_config_path = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    # groundingdino_config_path = "/home/hpc/Desktop/LLaVA-Det-Seg/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    groundingdino_checkpoint_path = "checkpoints/groundingdino_swint_ogc.pth" # 确保这个文件存在
    # groundingdino_checkpoint_path = "/home/hpc/Desktop/GroundingDino-Finetuning/fine_tuning_output/checkpoint0000.pth"
    sam_model_type = "vit_h"
    sam_checkpoint_path = "checkpoints/sam_vit_h_4b8939.pth" # 确保这个文件存在
    # 实例化模型
    model = LDSModel(
        llava_model_path=llava_model_path,
        groundingdino_config_path=groundingdino_config_path,
        groundingdino_checkpoint_path=groundingdino_checkpoint_path,
        sam_model_type=sam_model_type,
        sam_checkpoint_path=sam_checkpoint_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ).cuda()
    while True:
        # 准备输入数据
        # /home/hpc/Desktop/GroundingDino-Finetuning/DATASET/tinyperson/val/rgb/rgb_x0048/baidu_P000_timg (17).jpg
        image_path = input("请输入图片路径：")  # /home/hpc/Desktop/LLaVA-Det-Seg/.asset/basketball-2.png
        if not os.path.exists(image_path):
            print("File not found in {}".format(image_path))    
            continue
        question = input("请输入prompt：")  # where is the basketball?
        image_clip = load_clip_image(image_path)
        question = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images_tensor = process_images(
                image_clip,
                model.image_processor,
                model.kwargs
            ).to(model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, model.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        from groundingdino.util.inference import load_image, annotate

        image_source, image_det = load_image(image_path)
        with torch.inference_mode():
            # 运行模型
            llava_output, groundingdino_output, sam_output = model.eval_forward(images_tensor, input_ids, image_source, image_det)
        # 输出LLaVA对话
        print(llava_output)
        if groundingdino_output != {}:
            # 清空文件夹
            boxes_dir = "/home/hpc/Desktop/LLaVA-Det-Seg/outputs/boxes"
            masks_dir = "/home/hpc/Desktop/LLaVA-Det-Seg/outputs/masks"
            # 清空 boxes_dir（保留文件夹）
            for filename in os.listdir(boxes_dir):
                file_path = os.path.join(boxes_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # 删除文件或符号链接
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # 删除子目录
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

            # 清空 masks_dir（保留文件夹）
            for filename in os.listdir(masks_dir):
                file_path = os.path.join(masks_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

            # 颜色列表也使用 RGB [R, G, B]
            colors_rgb = [
                [255, 0, 0],   # 红色
                [0, 255, 0],   # 绿色
                [0, 0, 255],   # 蓝色
                [255, 255, 0], # 黄色
                [255, 0, 255], # 紫色
                [0, 255, 255], # 青色
                [255, 165, 0], # 橙色
                [128, 0, 128], # 紫色
            ]
            # 绘制检测框(outputs)
            annotated_frame = annotate(image_source=image_source, boxes=groundingdino_output["boxes"], logits=groundingdino_output["logits"], phrases=groundingdino_output["phrases"])
            save_frame_path = "{}/{}/{}_frame_img.jpg".format("outputs", "boxes", image_path.split("/")[-1].split(".")[0])
            print("{} 已保存.".format(save_frame_path))
            cv2.imwrite(save_frame_path, annotated_frame)
            # 绘制掩码
            all_save_img = image_source.copy()
            for i, mask in enumerate(sam_output["masks"]):
                if mask.shape[0] == 0:
                    continue

                mask = mask.detach().cpu().numpy()[0]
                mask = mask > 0

                # 从颜色列表中选择当前掩码的颜色
                # 使用索引 i 和模运算符 (%) 来循环使用颜色列表中的颜色
                current_color = colors_rgb[i % len(colors_rgb)]

                save_path = "{}/{}/{}_mask_{}.jpg".format(
                    "outputs", "masks", image_path.split("/")[-1].split(".")[0], i
                )
                cv2.imwrite(save_path, mask * 100)
                print("{} 已保存.".format(save_path))

                save_path = "{}/{}/{}_masked_img_{}.jpg".format(
                    "outputs", "masks", image_path.split("/")[-1].split(".")[0], i
                )
                # save_img = image_source.copy()
                # save_img[mask] = (
                #     image_source * 0.5
                #     + mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
                # )[mask]
                all_save_img[mask] = (
                    image_source * 0.5
                    + mask[:, :, None].astype(np.uint8) * np.array(current_color) * 0.5
                )[mask]
                # save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
                # cv2.imwrite(save_path, save_img)
                # print("{} 已保存.".format(save_path))
            # 保存总分割图片
            save_path = "{}/{}/{}_finalmask_img.jpg".format(
                    "outputs", "masks", image_path.split("/")[-1].split(".")[0]
                )
            all_save_img = cv2.cvtColor(all_save_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, all_save_img)
            print("{} 已保存.".format(save_path))



if __name__ == "__main__":
    main(sys.argv[1:])
