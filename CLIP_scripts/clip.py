from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# -------------------------
# CONFIG / HYPERPARAMS
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
FLAG_THRESHOLD = 0.4        # fraction of attributes that must be mismatched to flag
TOP_K_CLIP = 1              # top-k CLIP candidates to pick per attribute group

# -------------------------
# Controlled Vocab (dropdown options)
# -------------------------
CANDIDATES = {
    "category": ["t-shirt", "shirt", "kurta", "saree", "dress", "jeans", "trousers", "shorts", "shoes", "sneakers", "bag"],
    "color": ["Aqua Blue","Beige","Black","Blue","Brown","Coral","Cream","Gold","Green","Grey","Grey Melange","Khaki","Lavendar","Lemon Yellow","Maroon","Metallic","Mint Green","Multicolor","Mustard","Navy Blue","Nude","Olive","Orange","Peach","Pink","Purple","Red","Rust","Silver","Teal","White","Yellow"],
    # "pattern": ["Abstract","Animal","Aop","Aztec","Back Print","Bohemian","Botanical","Buffalo Checks","Camouflage","Cartoons","Checked","Chevron","Colorblocked","Conversational","Cricket","Ditsy Print","Dyed/ Washed","Embellished","Embroidered","Engineered Stripes","Ethnic Motif","Faded","Floral","Geometric","Gingham Checks","Goa","Graphic Print","Grid Tattersail Checks","Heathered","Horizontal Stripes","Houndstooth","Independence Day/Indian Flag","Mahakal","Micro Check","Micro Print","Newspaper","Ombre","Paisley","Pinstripes","Placement Print","Political Print","Polka Dots","Printed","Quirky","Religious Print","Self-Design","Solid","Striped","Superheroes","Tartan Checks","Tie & Dye","Tribal","Typography","Vertical Stripes","Windowpane Checks","Woven Design"],
    # "fabric": ["Acrylic","Art Silk","Bamboo","Chambray","Chanderi Cotton","Chanderi Silk","Chiffon","Cotton","Cotton Blend","Cotton Cambric","Cotton Linen","Cotton Silk","Crepe","Denim","Dupion Silk","Elastane","Elastodiene","Elastolefin","Georgette","Grey","Jacquard","Jute Cotton","Jute Silk","Khadi Cotton","Kora Muslin","Lace","Leather","Linen","Lycra","Lyocell","Modal","Mulmul","Net","Nylon","Organza","Paper Silk","Pashmina","Poly Blend","Poly Chiffon","Poly Crepe","Poly Georgette","Poly Silk","Polycotton","Polyester","Polypropylene","Popcorn","Rayon","Rayon Slub","Satin","Shantoon","Silk","Silk Blend","Soft Silk","Super Net","Synthetic","Taffeta Silk","Tissue","Tussar Silk","Velvet","Vichitra Silk","Viscose","Viscose Rayon","Voile","Wool"],
    "sleeve": ["Long Sleeves","Short Sleeves","Sleeveless","Three-Quater Sleeves"],
    "Pockets": ["Yes","No"],
    # "Closure":["Asymmetrical","Symmetric"],
}

# -------------------------
# Models initialization
# -------------------------
_clip_model: Optional[CLIPModel] = None
_clip_processor: Optional[CLIPProcessor] = None

def load_models():
    global _clip_model, _clip_processor
    if _clip_model is None:
        print(f"[INFO] Loading CLIP ({CLIP_MODEL_NAME}) on {DEVICE} ...")
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

# -------------------------
# CLIP-based zero-shot attribute extraction
# -------------------------
def clip_zero_shot_image_attributes(image: Image.Image,
                                    candidates: Dict[str, List[str]],
                                    top_k:int = TOP_K_CLIP
                                   ) -> Dict[str, List[Tuple[str, float]]]:
    if _clip_model is None or _clip_processor is None:
        raise RuntimeError("CLIP models not loaded. Call load_models() first.")
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
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        idxs = np.argsort(-probs)[:top_k]
        results[attr_name] = [(labels[i], float(probs[i])) for i in idxs]
    return results

# -------------------------
# High-level matching & flagging (exact match)
# -------------------------
def match_and_flag(image_path: str,
                   seller_attrs: Dict[str, str],
                   candidates: Dict[str, List[str]] = CANDIDATES,
                   flag_threshold: float = FLAG_THRESHOLD
                  ) -> Dict:
    """
    Returns:
      - extracted attributes
      - mismatch details
      - whether product is flagged
    """
    load_models()
    image = Image.open(image_path).convert("RGB")

    # Extract attributes from image
    img_attrs = clip_zero_shot_image_attributes(image, candidates, top_k=1)
    img_top = {k: (v[0][0] if v and len(v) > 0 else "") for k, v in img_attrs.items()}

    # Compare with seller-entered attributes (dropdowns, so exact match only)
    mismatches = 0
    total = len(candidates)
    results = {}

    for attr in candidates.keys():
        img_label = img_top.get(attr, "")
        seller_label = seller_attrs.get(attr, "")
        is_match = (img_label.lower() == seller_label.lower())  # strict match
        results[attr] = {"img_label": img_label, "seller_label": seller_label, "match": is_match}
        if not is_match:
            mismatches += 1

    mismatch_ratio = mismatches / total if total > 0 else 0
    flagged = mismatch_ratio >= flag_threshold

    return {
        "image_path": image_path,
        "img_top_attributes": img_top,
        "comparisons": results,
        "mismatch_ratio": mismatch_ratio,
        "flagged": flagged
    }

if _name_ == "_main_":
    # NOTe : replace with actual image path and seller attributes from your UI/form
    example_image = "/content/81aXoB3V3FL.UY1100.jpg"   # put a sample file next to this script
    example_seller_attrs = {
        "category": "shirt",
        "color": "Beige",
        "Pockets":"No",
        "Closure":"Symmetric",
        "sleeve": "Three-Quarter Sleeves"
    }

    load_models()
    # ensure SBERT model is initialized
    if _sbert_model is None:
        _sbert_model = SentenceTransformer(SBERT_MODEL)

    try:
        result = match_and_flag(example_image, example_seller_attrs)
        import json
        print(json.dumps(result, indent=2))
    except FileNotFoundError:
        print("[WARN] Example image not found. Please set example_image to a valid file path to test.")