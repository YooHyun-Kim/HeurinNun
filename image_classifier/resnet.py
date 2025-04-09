import torchvision.models as models
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image


def build_resnet(feature_only=False):
    """
    ResNet50 기반 분류기 생성. feature_only=True인 경우, 마지막 fc layer 제거.
    """
    model = models.resnet50(pretrained=True)
    if feature_only:
        model = nn.Sequential(*list(model.children())[:-1])  # avgpool까지 출력
    else:
        model.fc = nn.Identity()  # fc 없애고 feature vector로 사용 가능
    return model


def preprocess_image(img):  # PIL.Image.Image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(img).unsqueeze(0)  # (1, C, H, W)



def get_image_features(img_tensor, model, device="cpu"):
    """
    이미지 텐서를 받아서 feature vector 추출
    """
    model.eval()
    model.to(device)
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        features = model(img_tensor)
        features = features.squeeze()
        if len(features.shape) == 3:  # (C, 1, 1) 형태라면 펴주기
            features = features.view(features.size(0))
    return features.cpu().numpy()


# 예시용 코드
if __name__ == "__main__":
    model = build_resnet(feature_only=True)
    img_tensor = preprocess_image("example.jpg")
    features = get_image_features(img_tensor, model)
    print("[이미지 특징 벡터 추출 완료] → shape:", features.shape)