
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

# prompt = "Please identify all person in the image and provide their coordinates"
prompt = "Where is the basketball ?"
llava_path = "/home/hpc/Desktop/LLaVA/checkpoints/llava-v1.5-7b"
image_file = ".asset/basketball-2.png"

# IMAGE_PATH = ".asset/cat_dog.jpeg"
# TEXT_PROMPT = "object under the dog and cat"



args = type('Args', (), {
    "model_path": llava_path,
    "model_base": None,
    "model_name": get_model_name_from_path(llava_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)
