import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch


def build_densenet(num_classes: int = 5, feature_only: bool = False) -> nn.Module:
    """
    DenseNet121 기반 모델 생성 함수
    - feature_only=True: classification head 제거 후 feature extractor로 사용
    - feature_only=False: num_classes 분류기(head) 추가
    """
    model = models.densenet121(pretrained=True)
    if feature_only:
        # classifier 대신 Identity로 대체
        model = nn.Sequential(*list(model.features.children()),
                              nn.ReLU(inplace=True),
                              nn.AdaptiveAvgPool2d((1, 1)))
    else:
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    return model


def preprocess_image(img: Image.Image) -> torch.Tensor:
    """
    PIL 이미지 입력 -> 텐서 변환 및 정규화
    반환 형식: (1, C, H, W)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(img).unsqueeze(0)


def get_image_features(img_tensor: torch.Tensor, model: nn.Module, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    이미지 텐서를 받아 DenseNet으로부터 feature vector 추출
    - feature_only=True로 빌드된 경우, (batch, C, 1, 1) -> (C)
    - classification head가 있는 경우, 클래스 logits 반환
    """
    model.eval()
    model.to(device)
    with torch.no_grad():
        x = img_tensor.to(device)
        feats = model(x)
        # feature_only extractor일 경우, (B, C,1,1)
        if isinstance(feats, torch.Tensor) and feats.dim() == 4:
            feats = feats.view(feats.size(1))  # (C)
        else:
            feats = feats.squeeze()  # (num_classes) or scalar
    return feats.cpu()


# 예시 사용법
if __name__ == '__main__':
    # feature extractor로 사용
    model_feat = build_densenet(feature_only=True)
    img = Image.open('example.jpg').convert('RGB')
    tensor = preprocess_image(img)
    features = get_image_features(tensor, model_feat, device=torch.device('cpu'))
    print('Feature shape:', features.shape)

    # 분류기로 사용
    model_cls = build_densenet(num_classes=5, feature_only=False)
    logits = get_image_features(tensor, model_cls, device=torch.device('cpu'))
    print('Logits shape:', logits.shape)
