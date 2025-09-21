import openai
from typing import Dict
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoProcessor, LlavaForConditionalGeneration

# -------------------------
# CONFIG
# -------------------------
DEVICE = "cpu"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
FLAG_THRESHOLD = 0.4
TOP_K_CLIP = 1

# Candidates for dropdown
CANDIDATES = {
    "category": ["t-shirt","shirt","kurta","saree","dress","jeans","trousers","shorts","shoes","sneakers","bag"],
    "color": ["Aqua Blue","Beige","Black","Blue","Brown","Coral","Cream","Gold","Green","Grey","Grey Melange","Khaki","Lavendar","Lemon Yellow","Maroon","Metallic","Mint Green","Multicolor","Mustard","Navy Blue","Nude","Olive","Orange","Peach","Pink","Purple","Red","Rust","Silver","Teal","White","Yellow"],
    "sleeve": ["Long Sleeves","Short Sleeves","Sleeveless","Three-Quater Sleeves"],
    "Pockets": ["Yes","No"],
}

# -------------------------
# Load CLIP
# -------------------------
_clip_model, _clip_processor = None, None
def load_models():
    global _clip_model, _clip_processor
    if _clip_model is None:
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

def clip_zero_shot_image_attributes(image: Image.Image,
                                    candidates: Dict[str, list],
                                    top_k:int = TOP_K_CLIP) -> Dict[str, list]:
    results = {}
    for attr_name, labels in candidates.items():
        if attr_name == "category":
            captions = [f"a photo of a {lbl}" for lbl in labels]
        elif attr_name == "color":
            captions = [f"{lbl} color" for lbl in labels]
        else:
            captions = labels

        inputs = _clip_processor(text=captions, images=image, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = _clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]

        idxs = np.argsort(-probs)[:top_k]
        results[attr_name] = [(labels[i], float(probs[i])) for i in idxs]
    return results

# -------------------------
# Comparison: Image vs Seller attributes
# -------------------------
def compare_image_seller(image_path: str, seller_attrs: Dict[str, str]):
    load_models()
    image = Image.open(image_path).convert("RGB")

    img_attrs = clip_zero_shot_image_attributes(image, CANDIDATES, top_k=1)
    img_top = {k: (v[0][0] if v else "") for k,v in img_attrs.items()}

    mismatches = []
    for attr in CANDIDATES.keys():
        if img_top.get(attr,"").lower() != seller_attrs.get(attr,"").lower():
            mismatches.append(attr)

    return {
        "img_top_attributes": img_top,
        "mismatched_attributes": mismatches,
        "flagged": (len(mismatches)/len(CANDIDATES)) >= FLAG_THRESHOLD
    }
    
def image_enhanccer_tool(image_path: str)-> str:
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)

    # Optionally load the model in 4-bit for reduced memory usage
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_4bit=False # or load_in_8bit=True
    )
    
    lora_adapter_path = "checkpoint-3540/checkpoint-3540" # Replace with the actual path to your saved checkpoint
    model = PeftModel.from_pretrained(model, lora_adapter_path)
    model = model.merge_and_unload() 

    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    prompt = "<image>\npoint out some problems related to the quality of image which can be improved based on product's look"

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=100)

    # Decode the output
    generated_text = processor.decode(output[0], skip_special_tokens=True)
    return generated_text

# -------------------------
# Comparison: Seller attributes vs Seller description (LLM)
# -------------------------
def compare_seller_description(seller_attrs: Dict[str,str], description: str) -> dict:
    # system_prompt = """You are an assistant ... (same rules from updated_user_des.py)"""
    system_prompt = """
You are an assistant whose task is to compare product attributes found by a vision agent with the seller's description and report any mismatches.

Inputs:
1) Vision agent attributes are provided as a single-line or multi-line string (attribute names vary by product).
   Example tokens: color, pattern, fabric, fit, sleeve, closure, no_of_pockets, Blouse, Blouse Fabric, Net Quantity, etc.
   NOTE: attribute values can be empty strings or values like "No Blouse", "Without Blouse", "None", "N/A" to indicate absence.

2) Description: free-form seller text.

Rules (strict):
- Treat the following (case-insensitive) as **absence indicators** when they appear as attribute values in the vision agent: "no", "none", "without", "no blouse", "no_b blouse", "n/a", "na", "not included", "no blouse piece".
- Treat the description as **indicating presence** if it contains tokens/phrases like "blouse", "blouse piece", "unstitched blouse", "stitched blouse", "comes with blouse", "paired with * blouse", "with matching blouse", "includes blouse", or synonyms. Use substring matching and be case-insensitive; allow minor punctuation/spacing differences.
- If vision agent indicates **absent** (per absence indicators above) but the description mentions a corresponding item (e.g., blouse), that is a **mismatch**.
- If vision agent gives a value (e.g., color=Blue) and the description explicitly states a different value (e.g., "red"), that is a **mismatch**.
- If an attribute exists in only one source (vision OR description) and the other side makes no claim, **ignore** it (do not count as mismatch).
- For numeric/quantity attributes (e.g., "Net Quantity (N) : Single", "no_of_pockets: 2"), parse numeric words if obvious and compare (e.g., "single" -> 1). If parsing is ambiguous, do not declare mismatch.
- The assistant must only return a single JSON object (no extra text, no explanation). Use valid JSON.

Output JSON format (exact keys):
{
  "match": true|false,                     // boolean JSON true/false
  "mismatched_attributes": ["attr1", ...],// list of attribute names that mismatched (empty list if match = true)
  "reason": "detailed reason or 'All attributes match'"
}

When producing the reason, be specific (mention attribute names, vision values and description snippets that caused the mismatch).
"""

    # Flatten seller_attrs into "vision agent input" style string
    vision_str = ", ".join([f"{k} : {v}" for k,v in seller_attrs.items()])
    user_input = f"""
vision agent input: {vision_str}

description provided by seller: {description}
"""

    client = openai.AzureOpenAI(
        api_version="2024-06-01",
        azure_endpoint="https://genai-nexus.int.api.corpinter.net/apikey/",
        api_key="06c9643f-90c3-4b7a-8560-5b5e5a0d2730",
    )

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"system","content":system_prompt},
                  {"role":"user","content":user_input}],
        response_format={"type":"json_object"}
    )
    return completion.choices[0].message.content

# -------------------------
# Unified Pipeline
# -------------------------
def full_pipeline(image_path: str, seller_attrs: Dict[str,str], seller_description: str):
    img_vs_seller = compare_image_seller(image_path, seller_attrs)
    seller_vs_desc = compare_seller_description(seller_attrs, seller_description)
    img_enhance_suggestions = image_enhanccer_tool(image_path)
    return {
        "image_vs_seller": img_vs_seller,
        "seller_vs_description": seller_vs_desc,
        "image_enhancement_suggestions": img_enhance_suggestions
    }

# Example
if __name__ == "__main__":
    seller_input = {
        "category": "shirt",
        "color": "Beige",
        "Pockets":"No",
        "sleeve": "Three-Quarter Sleeves"
    }
    desc = "This is a beige shirt with three-quarter sleeves and a front pocket"
    result = full_pipeline("bba8b_512.jpg", seller_input, desc)
    print(result)
