import gradio as gr
import numpy as np
from PIL import Image
import os
import torch
import cv2
import requests
from io import BytesIO
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
from PIL import Image
import requests
import sys
import re
import os
# === 从你的项目中导入必要的模块和函数 ===
# 你需要确保以下导入是正确的，并且对应的文件或库已安装和可用
# 假设 LDSModel, disable_torch_init, process_images, tokenizer_image_token,
# conv_templates, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX 等都在你的项目路径下可导入
# 例如:
# from your_model_file import LDSModel, disable_torch_init, process_images, tokenizer_image_token
# from your_utils import conv_templates, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
# from groundingdino.util.inference import load_image, annotate # 确保 groundingdino 库已正确安装和配置
# from transformers import AutoTokenizer, AutoModelForCausalLM # LLaVA相关的可能导入
# from some_config import conv_templates # 假设 conv_templates 在某个配置文件中

# 模拟导入不存在的模块，你需要替换成你实际的导入
try:
    # --- 假设这是你项目的结构，请根据实际情况调整 ---
    from LDS import LDSModel # 假设 LDSModel 在当前包的 model.py 中
    from llava.utils import disable_torch_init # 假设一些工具函数在 utils.py 中
    from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    ) # 假设图像和文本处理函数在 process.py 中
    # 假设 conv_templates, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX 在一个 config 文件或 utils 文件中
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    # --- GroundingDINO 和 SAM 的导入，确保这些库已安装 ---
    from groundingdino.util.inference import load_image, annotate # GroundingDINO inference utils
    # 可能还需要其他SAM相关的导入，取决于你的LDSModel内部实现
    # from segment_anything import sam_model_registry, SamPredictor # 示例SAM导入

    print("Custom modules imported successfully.")

except ImportError as e:
    print(f"Warning: Failed to import custom modules. Please ensure your project structure and imports are correct. Error: {e}")
    print("Using placeholder classes/functions. The app will run but model processing will be skipped.")

    # === 提供占位符，以便代码结构可以运行，即使模型代码不可用 ===
    # 如果你的导入失败，这些占位符会允许 Gradio 界面启动，但处理逻辑是假的
    class LDSModel:
        def __init__(self, *args, **kwargs):
            print("Placeholder LDSModel initialized.")
            self.device = kwargs.get("device", "cpu")
            self.image_processor = type("MockImageProcessor", (), {"preprocess": lambda x: x})() # Mock processor
            self.tokenizer = type("MockTokenizer", (), {"pad_token_id": 0})() # Mock tokenizer
            self.kwargs = {} # Mock kwargs for process_images
        def cuda(self): return self
        def eval(self): return self
        def eval_forward(self, images_tensor, input_ids, image_source, image_det):
            print("Placeholder eval_forward called.")
            # 返回模拟输出
            llava_output = "Placeholder LLaVA output based on text."
            groundingdino_output = {"boxes": [], "logits": [], "phrases": []} # No detections
            sam_output = {"masks": []} # No masks
            return llava_output, groundingdino_output, sam_output

    def disable_torch_init(): print("Placeholder disable_torch_init called.")
    def process_images(image, processor, kwargs): print("Placeholder process_images called."); return torch.randn(1, 3, 224, 224) # 返回模拟张量
    def tokenizer_image_token(prompt, tokenizer, image_token_index, return_tensors="pt"): print("Placeholder tokenizer_image_token called."); return torch.randint(0, 100, (1, 10)) # 返回模拟张量
    conv_templates = {"llava_v1": type("MockConv", (), {"copy": lambda: type("MockConvInstance", (), {"append_message": lambda r, m: None, "get_prompt": lambda: "mock prompt", "roles": ["user", "assistant"]})()})()} # 模拟对话模板
    DEFAULT_IMAGE_TOKEN = "<image>"
    IMAGE_TOKEN_INDEX = -200

    # 模拟 GroundingDINO 和 SAM 导入的函数
    def load_image(image_path):
        print(f"Placeholder load_image called with {image_path}")
        if image_path is None: return None, None
        # 返回模拟的图像源和检测用图像
        img = cv2.imread(image_path) # OpenCV 读取是 BGR
        if img is None:
             # 创建一个空白图像作为占位符
            img = np.zeros((512, 512, 3), dtype=np.uint8)
            print(f"Warning: Could not load image {image_path}. Using placeholder.")
        img_source = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 模拟 RGB 源图像
        img_det = img # 检测用图像 (BGR)
        return img_source, img_det

    def annotate(image_source, boxes, logits, phrases):
        print("Placeholder annotate called.")
         # 返回一个空白图像作为占位符
        h, w, _ = image_source.shape if image_source is not None else (512, 512, 3)
        annotated_frame = np.zeros((h, w, 3), dtype=np.uint8) # 返回 BGR 格式
        return annotated_frame


    # 模拟其他可能需要的导入
    # from transformers import AutoTokenizer # 示例导入

# 加载 CLIP 图像的函数 
def load_clip_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

# === 全局变量，用于存储加载的模型和处理器 ===
# 这些变量会在应用启动时加载，并在 process_image_and_text 函数中使用
global_model = None
global_tokenizer = None
global_image_processor = None
global_device = "cuda" if torch.cuda.is_available() else "cpu" # 在加载模型前确定 device

# === 加载模型的函数 (在 Gradio 启动前调用一次) ===
def load_models():
    global global_model, global_tokenizer, global_image_processor, global_device
    print(f"Loading models on device: {global_device}...")

    # === 从 main(args) 函数中复制模型加载逻辑 ===
    # 模型路径和名称
    llava_model_path = "checkpoints/llava-LDS-v3"
    groundingdino_config_path = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    # groundingdino_checkpoint_path = "/home/hpc/Desktop/GroundingDino-Finetuning/fine_tuning_output/checkpoint0000.pth" 
    groundingdino_checkpoint_path = "checkpoints/groundingdino_swint_ogc.pth"
    sam_model_type = "vit_h"
    sam_checkpoint_path = "checkpoints/sam_vit_h_4b8939.pth" 

    # 调用 disable_torch_init() (如果需要)
    # try:
    #     disable_torch_init() # 如果你的 disable_torch_init 需要在模型加载前调用
    # except NameError:
    #     print("Warning: disable_torch_init not found, skipping.")
    #     pass # 如果 disable_torch_init 不存在，跳过

    kwargs = {} # 根据你的 parse_args 设置 kwargs，或者直接在这里定义
    # 例如，如果 image_aspect_ratio 需要传递给模型或处理器
    # kwargs['image_aspect_ratio'] = "square" # 示例

    try:
        # 实例化模型
        global_model = LDSModel(
            llava_model_path=llava_model_path,
            groundingdino_config_path=groundingdino_config_path,
            groundingdino_checkpoint_path=groundingdino_checkpoint_path,
            sam_model_type=sam_model_type,
            sam_checkpoint_path=sam_checkpoint_path,
            device=global_device,
            **kwargs # 传递 kwargs
        )
        if global_device == "cuda":
            global_model.cuda()
        global_model.eval() # 设置为评估模式

        # 获取 tokenizer 和 image_processor (取决于你的 LDSModel 如何提供它们)
        # 假设它们是模型的属性
        global_tokenizer = global_model.tokenizer
        global_image_processor = global_model.image_processor
        # === 复制结束 ===

        print("Models loaded successfully.")

    except Exception as e:
        print(f"Error loading models: {e}")
        print("Gradio app will start, but model processing will not work.")
        # 可以选择在这里 sys.exit(1) 如果模型加载失败应用就无法工作

# === Gradio 处理函数 ===
def process_image_and_text(image_file, text_input):
    """
    处理输入的图片和文本，调用加载好的模型流水线，并返回输出。

    Args:
        image_file: Gradio 提供的图片文件路径 (str) 或 None。
        text_input: 用户输入的文本 (str).

    Returns:
        一个包含三个元素的元组：(detection_image, segmentation_image, output_text)
        图像返回 numpy 数组 (RGB 格式)，文本返回字符串。
        如果没有图片输入或处理失败，相应的图片返回 None。
    """
    # 检查模型是否已加载
    if global_model is None or global_tokenizer is None or global_image_processor is None:
        return None, None, "Error: Models not loaded. Please check the server logs."

    output_text = ""
    detection_image = None
    segmentation_image = None
    llava_text_output = ""

    try:
        # === 从你原始 main 函数的 while 循环中复制处理逻辑 ===

        # 1. 处理无图片输入的情况
        if image_file is None:
            print("No image provided, only processing text.")
            # 如果没有图片，可能只运行 LLaVA 的文本部分
            # 这取决于你的 LDSModel 是否支持纯文本输入
            # 简单起见，这里只返回收到的文本和提示用户没有图片
            llava_text_output = f"Received text: {text_input}\nNote: No image provided, only text was considered."
            # 如果你的LDSModel支持纯文本LLaVA，可以在这里调用
            # 例如：
            conv_mode = "llava_v1" # 或者你的模型对应的conv mode
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], text_input) # 纯文本输入
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, global_tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(global_device)
            with torch.inference_mode():
                # 假设你的 eval_forward 能处理 images_tensor=None 或是一个 Dummy 张量
                # 或者你的 LDSModel 有一个专门的 text_only_forward 方法
                llava_output, _, _ = global_model.eval_forward(None, input_ids, None, None) # 假设这样可以传递 None 或 Dummy
            llava_text_output = llava_output

            # 没有图片输入，图片输出返回 None
            return None, None, llava_text_output

        # 2. 处理有图片输入的情况
        # 检查文件是否存在（Gradio上传的文件通常存在于临时路径，但可以在这里加一个检查）
        if not os.path.exists(image_file):
             llava_text_output = f"Error: Uploaded image file not found: {image_file}"
             return None, None, llava_text_output


        print(f"Processing image: {image_file} with text: {text_input}")

        # 加载 CLIP 图像 (用于 LLaVA image_processor)
        image_clip = load_clip_image(image_file)

        # 加载 GroundingDINO 和 SAM 用图像 (通常是 OpenCV 格式)
        # load_image 函数返回 RGB 格式的 image_source 和 BGR 格式的 image_det
        image_source_rgb, image_det_bgr = load_image(image_file)
        if image_source_rgb is None or image_det_bgr is None:
             llava_text_output = f"Error: Could not load image for detection/segmentation: {image_file}"
             return None, None, llava_text_output


        # 准备 prompt (根据你的 LLaVA 格式)
        # original_question = text_input # 保存原始文本
        question_with_token = DEFAULT_IMAGE_TOKEN + "\n" + text_input
        conv_mode = "llava_v1" # 根据你的模型确定对话模式
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], question_with_token)
        conv.append_message(conv.roles[1], None) # 模型生成的部分
        prompt = conv.get_prompt()


        # 预处理图像和文本
        images_tensor = process_images(
                image_clip,
                global_image_processor,
                global_model.kwargs # 使用全局模型的 kwargs
            ).to(global_device, dtype=torch.float16) # 确保数据类型和设备正确

        input_ids = (
            tokenizer_image_token(prompt, global_tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0) # 增加 batch 维度
            .to(global_device) # 放到正确的设备
        )


        # 运行模型推理
        with torch.inference_mode(): # 使用 inference_mode 提高效率和减少内存使用
            llava_output, groundingdino_output, sam_output = global_model.eval_forward(
                images_tensor,
                input_ids,
                image_source_rgb, # GroundingDINO/SAM 需要 RGB source
                image_det_bgr # GroundingDINO/SAM 需要 BGR for annotate (虽然 annotate 用 source) 或其他内部处理
            )

        # 3. 处理模型输出 (移除文件保存逻辑)

        # LLaVA 对话输出
        llava_text_output = llava_output
        print("LLaVA Output:", llava_text_output)

        # GroundingDINO 检测输出
        if groundingdino_output and groundingdino_output.get("boxes") is not None and len(groundingdino_output["boxes"]) > 0:
            # 使用 annotate 函数在 image_source_rgb 上绘制检测框
            # annotate 函数通常返回 BGR 格式的 NumPy 数组
            annotated_frame_bgr = annotate(
                image_source=image_source_rgb, # annotate 需要 RGB 图像源进行绘制
                boxes=groundingdino_output["boxes"],
                logits=groundingdino_output["logits"],
                phrases=groundingdino_output["phrases"]
            )
            # Gradio 显示通常需要 RGB，将 BGR 转换为 RGB
            detection_image = cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB)
            print("Generated detection image.")
        else:
             print("No objects detected by GroundingDINO.")
             detection_image = None # 如果没有检测结果，图片输出返回 None

        # SAM 分割输出
        if sam_output and sam_output.get("masks") is not None and len(sam_output["masks"]) > 0:
            # === 复制绘制掩码逻辑 ===
            all_save_img_rgb = image_source_rgb.copy() # 在 RGB 源图像上绘制掩码

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

            for i, mask in enumerate(sam_output["masks"]):
                # 确保 mask 是 numpy 数组且形状正确
                if isinstance(mask, torch.Tensor):
                     mask = mask.detach().cpu().numpy()

                if mask.shape[0] == 0:
                    continue

                # 假设 mask 是 (H, W) 或 (1, H, W)，需要 (H, W) 的布尔数组
                if mask.ndim == 3 and mask.shape[0] == 1:
                     mask = mask[0]

                mask = mask > 0 # 转换为布尔掩码

                current_color = colors_rgb[i % len(colors_rgb)]

                # 在 all_save_img_rgb (RGB 格式) 上绘制半透明彩色掩码
                # 注意：这里的颜色数组 np.array(current_color) 已经是 RGB
                # image_source * 0.5 + mask[:, :, None] * color * 0.5 混合颜色
                # 确保掩码是 uint8 类型用于乘法
                mask_uint8 = mask[:, :, None].astype(np.uint8)
                all_save_img_rgb[mask] = (
                    image_source_rgb[mask] * 0.5 + mask_uint8[mask] * np.array(current_color) * 0.5
                ).astype(np.uint8) # 转换回 uint8

            segmentation_image = all_save_img_rgb # 最终的分割图像 (RGB NumPy)
            print("Generated segmentation image.")

        else:
            print("No masks generated by SAM.")
            segmentation_image = None # 如果没有分割结果，图片输出返回 None

        # === 复制处理逻辑结束 ===

    except Exception as e:
        # 捕获处理过程中的任何异常，并返回错误信息，避免 Gradio 崩溃
        error_message = f"An error occurred during processing: {e}"
        print(error_message)
        import traceback
        traceback.print_exc() # 打印详细错误信息到终端

        llava_text_output = f"Processing failed: {e}\nCheck server logs for details."
        detection_image = None
        segmentation_image = None


    # 返回结果，顺序与 Gradio outputs 对应
    # pre_saved_detection_image_path = "/home/hpc/Desktop/LLaVA-Det-Seg/outputs/good_outputs/rgb_0078_00000003_frame_img.jpg"  
    # pre_saved_segmentation_image_path = "/home/hpc/Desktop/LLaVA-Det-Seg/outputs/good_outputs/rgb_0078_00000003_finalmask_img.jpg" 
    # loaded_detection_image = Image.open(pre_saved_detection_image_path).convert("RGB")
    # loaded_segmentation_image = Image.open(pre_saved_segmentation_image_path).convert("RGB")
    # detection_image = loaded_detection_image
    # segmentation_image = loaded_segmentation_image
    # llava_output = "[DET] boat"
    return detection_image, segmentation_image, llava_text_output


# === 使用 gr.Blocks 进行更灵活的布局 (沿用之前的布局代码) ===
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 图像与文本处理演示
        上传图片（可选）并输入文本，进行检测、分割和自然语言处理。
        """
    )

    # 主体内容，左右布局
    with gr.Row():
        # 左侧：输入区域
        with gr.Column(scale=1, min_width=300): # 可以调整 scale 和 min_width
            input_image = gr.Image(
                type="filepath", # 返回图片文件路径
                label="选择图片文件 (可选)",
            )
            input_text = gr.Textbox(
                lines=3,
                label="输入自然语言提示",
                placeholder="在这里输入您的文本..."
            )
            # 提交按钮，设为 primary 样式 (通常是橙色)
            submit_btn = gr.Button("提交", variant="primary")
            # 清除按钮，设为 secondary 样式 (次要操作)
            clear_btn = gr.Button("清除", variant="secondary")


        # 右侧：输出区域
        with gr.Column(scale=2, min_width=500): # 输出区域可以更宽
            gr.Markdown("### 处理结果")
            with gr.Row(): # 将两个图像输出放在同一行
                # Gradio 的 Image 输出可以接收 NumPy 数组
                output_det_image = gr.Image(label="检测输出", interactive=False, format="png") # 可以指定格式
                output_seg_image = gr.Image(label="分割输出", interactive=False, format="png") # 可以指定格式
            output_text = gr.Textbox(label="自然语言输出", interactive=False, lines=5) # 增加行数以便显示更多文本

    # 绑定按钮的点击事件到处理函数
    # inputs 列表和 outputs 列表的顺序必须与 process_image_and_text 函数的参数和返回值顺序对应
    submit_btn.click(
        fn=process_image_and_text,
        inputs=[input_image, input_text],
        outputs=[output_det_image, output_seg_image, output_text]
    )
    # 当用户在 input_text 文本框中按下 Enter 键时，触发 process_image_and_text 函数
    input_text.submit(
        fn=process_image_and_text,
        inputs=[input_image, input_text],
        outputs=[output_det_image, output_seg_image, output_text]
    )
    # 清除按钮的功能：将所有输入和输出组件的值设为 None 或空字符串
    clear_btn.click(
        # 返回值顺序：input_image, input_text, output_det_image, output_seg_image, output_text
        fn=lambda: [None, "", None, None, ""],
        inputs=[], # 清除按钮不需要输入
        outputs=[input_image, input_text, output_det_image, output_seg_image, output_text]
    )


# === 主程序入口 ===
if __name__ == "__main__":
    # 在 Gradio 应用启动前，先加载模型
    load_models()
    # 启动 Gradio 应用
    print("Gradio 应用正在启动...")
    print("请在浏览器中打开显示的本地地址 (通常是 http://127.0.0.1:7860 或其他端口)")
    demo.launch()