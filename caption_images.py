from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os
import re

processed_path = "/content/drive/MyDrive/LoRA_Project/images/processed"
captioned_path = "/content/drive/MyDrive/LoRA_Project/images/captioned"
TRIGGER_WORD = "ohwx man"

os.makedirs(captioned_path, exist_ok=True)

# Load BLIP
print("Loading BLIP model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()
print("BLIP ready!")

# Caption all images
images = [f for f in os.listdir(processed_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"Captioning {len(images)} images...")

for filename in images:
    img_path = os.path.join(processed_path, filename)
    img = Image.open(img_path).convert("RGB")

    inputs = processor(img, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)

    raw_caption = processor.decode(output[0], skip_special_tokens=True)
    final_caption = f"{TRIGGER_WORD}, {raw_caption}"

    txt_filename = os.path.splitext(filename)[0] + ".txt"
    txt_path = os.path.join(captioned_path, txt_filename)

    with open(txt_path, "w") as f:
        f.write(final_caption)

    img.save(os.path.join(captioned_path, filename))
    print(f"Captioned: {filename}: {final_caption}")

# Clean noisy captions
noise_patterns = [
    r"with the word.*",
    r"with a.*logo.*",
    r"grey hair",
    r"sunglasses",
]

txt_files = [f for f in os.listdir(captioned_path) if f.endswith(".txt")]
for txt_file in txt_files:
    path = os.path.join(captioned_path, txt_file)
    with open(path, "r") as f:
        caption = f.read()
    for pattern in noise_patterns:
        caption = re.sub(pattern, "", caption)
    caption = caption.replace("sunglasses", "glasses")
    caption = re.sub(r",\s*,", ",", caption)
    caption = re.sub(r"\s+", " ", caption).strip()
    caption = caption.rstrip(",").strip()
    with open(path, "w") as f:
        f.write(caption)

print("All captions cleaned!")
