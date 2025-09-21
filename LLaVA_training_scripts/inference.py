import torch
from peft import PeftModel
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image


model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)

# Optionally load the model in 4-bit for reduced memory usage
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    load_in_4bit=False, # or load_in_8bit=True,
    device_map="cpu"
)

# 2. Load the trained PEFT adapters
lora_adapter_path = "checkpoint-3540/checkpoint-3540" # Replace with the actual path to your saved checkpoint
model = PeftModel.from_pretrained(model, lora_adapter_path)
model = model.merge_and_unload() 

image_path = "bba8b_512.jpg" # Replace with the path to the image you want to test
image = Image.open(image_path).convert("RGB")
image = image.resize((224, 224))
prompt = "<image>\npoint out some problems related to the quality of image which can be improved based on product's look"

inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=50)

# Decode the output
generated_text = processor.decode(output[0], skip_special_tokens=True)
print(generated_text)

