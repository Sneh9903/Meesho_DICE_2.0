import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from trl import SFTTrainer
from datasets import load_dataset

from trl import SFTTrainer, SFTConfig 


import json
from PIL import Image
from datasets import Dataset

import torch
print(torch.cuda.is_available())

# 1. Load your JSON file
with open("train_data_final.json", "r") as f:
    raw_data = json.load(f)

# 2. Create a list of dictionaries with the required 'messages' format
processed_data = []
image_folder = "images"
i = 0
for entry in raw_data:
    # Assuming your JSON has keys like 'image_id' and 'output_text'
    img_id = entry['image']
    output_text = entry['text']
    
    # Load the image
    try:
        image_path = f"{image_folder}/{img_id}" # or whatever the file extension is
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))  # Resize to 224x224 pixels
        
        # Construct the conversation in the 'messages' format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "point out some problems related to the quality of image which can be improved based on product's look"},
                    {"type": "image", "image": image}
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": output_text}
                ],
            },
        ]
        
        processed_data.append({
            "messages": messages,
            "images": [image]   # ✅ add this
        })
    except:
        continue

# 3. Create the Hugging Face Dataset
dataset = Dataset.from_list(processed_data)



# Load the base model and processor
model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)

# Optionally load the model in 4-bit for reduced memory usage
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    load_in_4bit=False # or load_in_8bit=True
)

# Prepare the model for k-bit training (if using quantization)
# model = prepare_model_for_kbit_training(model)

# Define the LoRA configuration
peft_config = LoraConfig(
    r=16, # The rank of the update matrices
    lora_alpha=32, # LoRA scaling factor
    target_modules=["q_proj", "v_proj", "k_proj"], # The layers to apply LoRA to
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply the LoRA config to the model
model = get_peft_model(model, peft_config)

# Print the trainable parameters to see the significant reduction
model.print_trainable_parameters()


from trl import SFTTrainer, SFTConfig

# ✅ Put max_seq_length here inside SFTConfig
sft_config = SFTConfig(
    dataset_text_field="messages",
    max_length=2048,   # here, not in SFTTrainer
    output_dir="./llava_lora_results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    num_train_epochs=10,
    logging_steps=5,
    save_strategy="epoch",
    learning_rate=5e-5,
    bf16=False,
    fp16=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=sft_config,
    processing_class=processor,  # ✅ tell trainer how to handle text+images
)



# Start training!
trainer.train()