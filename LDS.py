import torch
import argparse
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForCausalLM
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN,IMAGE_PLACEHOLDER
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.conversation import conv_templates

import groundingdino.models.GroundingDINO.groundingdino as groundingdino_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util import *
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.inference import load_model, load_image, predict, annotate
from segment_anything import sam_model_registry
from segment_anything import SamPredictor # 方便后续使用 SAM 的 transform
from torchvision.ops import box_convert

import re

class LDSModel(nn.Module):
    def __init__(self, llava_model_path,
                 groundingdino_config_path, 
                 groundingdino_checkpoint_path,
                 sam_model_type, 
                 sam_checkpoint_path, 
                 device="cuda",
                 **kwargs,
        ):
        super().__init__()
        self.device = device
        self.kwargs = kwargs
        # 加载 LLaVA 模型
        model_name = get_model_name_from_path(llava_model_path)
        self.tokenizer, self.llava_model, self.image_processor, self.context_len = load_pretrained_model(model_path=llava_model_path, model_base=None, model_name=model_name)
        self.llava_model.to(self.device)

        # 加载 GroundingDINO 模型
        self.groundingdino_config_file = groundingdino_config_path
        self.groundingdino_checkpoint_path = groundingdino_checkpoint_path
        self.groundingdino_model = self._load_groundingdino().to(self.device)

        # 加载 SAM 模型
        self.sam_model_type = sam_model_type
        self.sam_checkpoint_path = sam_checkpoint_path
        self.sam_model = self._load_sam().to(self.device)
        self.sam_predictor = SamPredictor(self.sam_model) # 初始化 SamPredictor

        # self.post_init()
    
    # 加载groundingdino模型，设置为eval模式，不训练
    def _load_groundingdino(self):
        args = SLConfig.fromfile(self.groundingdino_config_file)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(self.groundingdino_checkpoint_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        return model

    # 加载SAM模型
    def _load_sam(self):
        sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint_path)
        return sam
    def forward(
        self,
        image_clip,
        input_ids,
        image_source,
        image_det,
        ground_truth_boxes=None,
        ground_truth_phrases=None,
        ground_truth_text=None, # 新增ground_truth_text用于LLaVA的训练
    ):
        """
        Forward function for training LLaVA and GroundingDINO.

        Args:
            image_clip (torch.Tensor): Image tensor for LLaVA.
            input_ids (torch.Tensor): Input token IDs for LLaVA.
            image_source (torch.Tensor): Original image tensor for SAM (not directly used in training here).
            image_det (torch.Tensor): Image tensor for GroundingDINO.
            ground_truth_boxes (list[list[float]], optional): Ground truth bounding boxes [x1, y1, x2, y2]. Defaults to None.
            ground_truth_phrases (list[str], optional): Ground truth object descriptions. Defaults to None.
            ground_truth_text (str, optional): Ground truth text containing object descriptions and "[DET]". Defaults to None.

        Returns:
            dict: Dictionary containing the loss values for LLaVA and GroundingDINO.
        """
        losses = {}

        # 1. LLaVA Forward Pass and Loss
        if ground_truth_text is not None:
            llava_output = self.llava_model(
                input_ids,
                images=image_clip,
                image_sizes=1,
                return_dict=True,
            )
            logits = llava_output.logits
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            llava_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            losses['llava_loss'] = llava_loss

            # 获取LLaVA的预测文本，用于GroundingDINO的输入
            with torch.no_grad():
                llava_output_ids = self.llava_model.generate(
                    input_ids,
                    images=image_clip,
                    image_sizes=1,
                    do_sample=False,
                    temperature=0,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=512,
                    use_cache=True,
                )
                predicted_text = self.tokenizer.batch_decode(llava_output_ids, skip_special_tokens=True)[0].strip()
                detected_objects_from_llava = None
                if "[DET]" in predicted_text:
                    detected_objects_from_llava = predicted_text.replace("[DET]", "").replace("Okay", "").replace("Sure", "").replace("Of course", "").strip()
                    detected_objects_from_llava = re.sub(r'\d+', '', detected_objects_from_llava).strip()
        else:
            detected_objects_from_llava = None

        # 2. GroundingDINO Forward Pass and Loss
        if ground_truth_boxes is not None and ground_truth_phrases is not None and detected_objects_from_llava:
            # 将ground truth boxes转换为GroundingDINO期望的格式 (normalized [cx, cy, w, h])
            target_boxes = torch.tensor(ground_truth_boxes, dtype=torch.float32, device=self.device)
            h, w = image_source.shape[:2]
            target_boxes = box_convert(target_boxes, in_fmt="xyxy", out_fmt="cxcywh")
            target_boxes[:, 0::2] /= w
            target_boxes[:, 1::2] /= h

            target_phrases = ground_truth_phrases

            outputs = self.groundingdino_model(image_det, detected_objects_from_llava)

            # 计算GroundingDINO的损失 (这里需要使用GroundingDINO模型内部定义的损失函数)
            # 你可能需要修改GroundingDINO模型的forward函数或者直接调用其loss函数（如果可用）
            # 这里假设模型有一个compute_loss方法，你需要根据实际情况调整
            try:
                groundingdino_loss = self.groundingdino_model.compute_loss(outputs, targets=[{"boxes": target_boxes, "labels": target_phrases}])
                losses['groundingdino_loss'] = groundingdino_loss
            except AttributeError:
                print("Warning: GroundingDINO model does not have a 'compute_loss' method directly accessible. You might need to implement the loss calculation based on the model's output.")
                # 如果没有compute_loss方法，你需要根据GroundingDINO的输出(pred_logits, pred_boxes)和ground truth(target_boxes, target_phrases)手动计算损失
                # 这通常涉及到分类损失（例如 Focal Loss）和框回归损失（例如 L1 Loss 或 GIoU Loss）
                pass # 在这里添加自定义的GroundingDINO损失计算逻辑

        return losses
    
    def eval_forward(
        self, 
        image_clip, 
        input_ids,
        image_source,
        image_det,
    ):
        # 1. LLaVA 的处理
        llava_output_ids = self.llava_model.generate(
            input_ids,
            images=image_clip,
            image_sizes=1,
            do_sample=False,
            temperature=0,
            top_p=None,
            num_beams=1,
            max_new_tokens=512,
            use_cache=True,
        )
        llava_text_output = self.tokenizer.batch_decode(llava_output_ids, skip_special_tokens=True)[0].strip()
        llava_text_output = llava_text_output.replace("\n", "").replace("  ", " ")
        # print(llava_text_output)
        # 2. 解析 LLaVA 的输出，获取 [DET] 和 [SEG] 指令以及可能的对象描述
        detected_objects = None
        
        if "[DET]" in llava_text_output:
            # 假设 [DET] 后面跟着需要检测的对象描述
            detected_objects = llava_text_output.replace("[DET]", "").replace("Okay", "").replace("Sure", "").replace("Of course", "")
            detected_objects = re.sub(r'\d+', '', detected_objects)


        # 3. GroundingDINO 的处理 (如果 LLaVA 输出包含 [DET])
        BOX_TRESHOLD=0.3
        TEXT_TRESHOLD=0.3
        groundingdino_output={}
        sam_output={}
        if detected_objects:
            boxes, logits, phrases = predict(
            model=self.groundingdino_model,
            image=image_det,
            caption=detected_objects,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
            )
            if boxes.shape[0] > 0:
                groundingdino_output={
                    "boxes": boxes,
                    "logits": logits,
                    "phrases": phrases,
                }

                # 4. SAM 的处理 (如果 LLaVA 输出包含 [SEG] 且 GroundingDINO 有输出)
                # sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint_path)
                # sam.to(device=self.device)
                # predictor = SamPredictor(sam)

                # groundingDINO原始输出boxes格式为：[cx, cy, w, h]
                # image_source的格式为：[H, W]（原始图像）
                
                boxes_for_sam = boxes.clone()
                boxes_for_sam[:, 0] = boxes_for_sam[:, 0] * image_source.shape[1]
                boxes_for_sam[:, 1] = boxes_for_sam[:, 1] * image_source.shape[0]
                boxes_for_sam[:, 2] = boxes_for_sam[:, 2] * image_source.shape[1]
                boxes_for_sam[:, 3] = boxes_for_sam[:, 3] * image_source.shape[0]
                boxes_for_sam = box_convert(boxes=boxes_for_sam, in_fmt="cxcywh", out_fmt="xyxy")
                # 分割任务
                self.sam_predictor.set_image(image_source)  # 生成图像embedding，以便让mask处理
                transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_for_sam, image_source.shape[:2])
                masks, mask_scores, mask_logits = self.sam_predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes.cuda(),
                    multimask_output=False,
                )
                sam_output = {
                    "masks": masks,
                    "mask_scores": mask_scores,
                    "mask_logits": mask_logits,
                }
            # 尝试调整一下llava_out的计数数目
            llava_text_output = re.sub(r'\d+', str(boxes.shape[0]), llava_text_output)
    
        return llava_text_output, groundingdino_output, sam_output
    
