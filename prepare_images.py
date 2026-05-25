from PIL import Image, ImageOps
import os

raw_path = "/content/drive/MyDrive/LoRA_Project/images/raw"
processed_path = "/content/drive/MyDrive/LoRA_Project/images/processed"
TARGET_SIZE = 512

os.makedirs(processed_path, exist_ok=True)

images = [f for f in os.listdir(raw_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"Found {len(images)} images\n")

for filename in images:
    input_path = os.path.join(raw_path, filename)
    output_path = os.path.join(processed_path, filename)

    img = Image.open(input_path).convert("RGB")
    
    img.thumbnail((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    
    padded = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))
    offset = ((TARGET_SIZE - img.width) // 2, (TARGET_SIZE - img.height) // 2)
    padded.paste(img, offset)
    
    padded.save(output_path, quality=95)
    print(f"✅ Processed: {filename}")

print(f"\n🎉 Done! {len(images)} images saved to processed/")
