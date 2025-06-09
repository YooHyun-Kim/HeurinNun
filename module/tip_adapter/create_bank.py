import os
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

def build_memory_bank(prototype_dir: str = "data/prototypes",
                      output_path: str = "memory_bank/memory_bank.pt",
                      clip_model_name: str = "openai/clip-vit-base-patch32",
                      device: torch.device = None):
    

    """
    Tip-Adapter용 메모리 뱅크 생성:
    prototype_dir 폴더 아래에 도메인별로 이미지가 라벨 폴더로 구분되어 있어야 합니다.
      prototype_dir/
        ├── 설계도/
        │     ├── img1.png
        │     └── img2.jpg
        ├── 회로도/
        │     ├── circ1.png
        │     └── circ2.jpg
        └── ...

    출력: keys, values, label2idx를 포함한 torch 파일 저장
    """


    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    model = CLIPModel.from_pretrained(clip_model_name).to(device)
    model.eval()

    keys = []
    values = []
    labels = sorted(os.listdir(prototype_dir))
    label2idx = {label: idx for idx, label in enumerate(labels)}

    for label in labels:
        class_dir = os.path.join(prototype_dir, label)
        for fname in os.listdir(class_dir):
            img_path = os.path.join(class_dir, fname)
            try:
                img = Image.open(img_path).convert("RGB")
            except:
                continue
            inputs = processor(images=img, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                feat = model.get_image_features(**inputs)  # [1, dim]
            keys.append(feat.cpu().squeeze(0))
            values.append(label2idx[label])

    keys = torch.stack(keys)       # [N, dim]
    values = torch.tensor(values)  # [N]
    torch.save({"keys": keys, "values": values, "label2idx": label2idx}, output_path)

    print(f"Memory bank created with {keys.size(0)} entries.")
    print(f"Saved to {output_path}")

# 사용 예시
if __name__ == "__main__":
    build_memory_bank(
        prototype_dir="data/prototypes",       
        output_path="memory_bank/memory_bank.pt"
    )
