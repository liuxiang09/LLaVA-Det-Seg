import argparse
import torch
import cv2
import matplotlib
from torchvision.ops import box_convert

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    import numpy as np
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    import cv2
    from segment_anything import sam_model_registry, SamPredictor
    import random
    
    # 初始化分割模型
    sam_checkpoint = "/home/hpc/Desktop/segment-anything/checkpoints/sam_vit_h_4b8939.pth"
    sam_model_type = "vit_h"
    device = model.device
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device=model.device)
    predictor = SamPredictor(sam)
    # predictor.model.to('cuda:0')

    # LLaVA模型输出
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)

    BOX_TRESHOLD = 0.5      # 检测框阈值
    TEXT_TRESHOLD = 0.25    # 文本阈值
    det_model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "/home/hpc/Desktop/GroundingDINO/checkpoints/groundingdino_swint_ogc.pth")

    image_source, image = load_image(args.image_file)
    # image_source = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
    
    # 生成检测框
    boxes, logits, phrases = predict(
        model=det_model,
        image=image,
        caption=outputs,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite("outputs/boxes/annotated_image.jpg", annotated_frame)
    # groundingDINO原始输出boxes格式为：[cx, cy, w, h]
    # image_source的格式为：[H, W]（原始图像）
    boxes[:, 0] = boxes[:, 0] * image_source.shape[1]
    boxes[:, 1] = boxes[:, 1] * image_source.shape[0]
    boxes[:, 2] = boxes[:, 2] * image_source.shape[1]
    boxes[:, 3] = boxes[:, 3] * image_source.shape[0]
    boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
    # 分割任务
    predictor.set_image(image_source)  # 生成图像embedding，以便让mask处理
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes, image_source.shape[:2])
    masks, mask_scores, mask_logits = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.cuda(),
        multimask_output=False,
    )

    # 绘制掩码
    for i, mask in enumerate(masks):
        if mask.shape[0] == 0:
            continue

        mask = mask.detach().cpu().numpy()[0]
        mask = mask > 0

        save_path = "{}/{}_mask_{}.jpg".format(
            "outputs", args.image_file.split("/")[-1].split(".")[0], i
        )
        cv2.imwrite(save_path, mask * 100)
        print("{} has been saved.".format(save_path))

        save_path = "{}/{}_masked_img_{}.jpg".format(
            "outputs", args.image_file.split("/")[-1].split(".")[0], i
        )
        save_img = image_source.copy()
        save_img[mask] = (
            image_source * 0.5
            + mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
        )[mask]
        save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, save_img)
        print("{} has been saved.".format(save_path))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
