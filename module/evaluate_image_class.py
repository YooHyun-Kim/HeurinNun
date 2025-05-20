import os
import sys
import argparse
from PIL import Image
import torch

# 프로젝트 루트를 PYTHONPATH에 포함
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# 모델별 build/import 함수
from image_classifier.resnet import build_resnet, preprocess_image as resnet_preprocess, get_image_features as resnet_get_features
from image_classifier.densenet import build_densenet, preprocess_image as densenet_preprocess, get_image_features as densenet_get_features
from module.tip_adapter.tip_adapter import predict_image_tip_adapter


def predict_resnet(img: Image.Image, model: torch.nn.Module, device: torch.device):
    tensor = resnet_preprocess(img).to(device)
    model.eval()
    with torch.no_grad():
        out = model(tensor)
        idx = out.argmax(dim=1).item()
    return idx


def predict_densenet(img: Image.Image, model: torch.nn.Module, device: torch.device):
    tensor = densenet_preprocess(img).to(device)
    model.eval()
    with torch.no_grad():
        out = model(tensor)
        idx = out.argmax(dim=1).item()
    return idx


def main():
    parser = argparse.ArgumentParser(description="이미지 분류기 평가 (ResNet/DenseNet/Tip-Adapter)")
    parser.add_argument('--model', choices=['resnet', 'densenet', 'tip_adapter'], required=True,
                        help='평가할 백본 모델')
    parser.add_argument('--test_dir', default=os.path.join('data','image_dataset','test'),
                        help='테스트 이미지가 있는 디렉터리')
    parser.add_argument('--resnet_ckpt', default=os.path.join('module','resnet_classifier.pth'),
                        help='(resnet 사용 시) ResNet 모델 체크포인트 경로')
    parser.add_argument('--densenet_ckpt', default=os.path.join('module','densenet_classifier.pth'),
                        help='(densenet 사용 시) DenseNet 모델 체크포인트 경로')
    args = parser.parse_args()

    # tip_adapter는 별도 pth 체크포인트 없이 memory_bank.pt를 내부에서 로드합니다.

    # 평가 디렉터리 및 클래스 리스트
    test_dir = args.test_dir
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    class_names = sorted([d for d in os.listdir(test_dir)
                          if os.path.isdir(os.path.join(test_dir, d))])
    num_classes = len(class_names)
    print(f"Detected classes: {class_names}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 로딩
    if args.model == 'resnet':
        model = build_resnet(num_classes=num_classes, feature_only=False).to(device)
        ckpt = args.resnet_ckpt
        model.load_state_dict(torch.load(ckpt, map_location=device))
        predictor = lambda img: predict_resnet(img, model, device)
    elif args.model == 'densenet':
        model = build_densenet(num_classes=num_classes, feature_only=False).to(device)
        ckpt = args.densenet_ckpt
        model.load_state_dict(torch.load(ckpt, map_location=device))
        predictor = lambda img: predict_densenet(img, model, device)
    else:
        predictor = lambda img: class_names.index(predict_image_tip_adapter(img)[0])

    # 평가 수행
    total = 0
    correct = 0
    for cls in class_names:
        cls_dir = os.path.join(test_dir, cls)
        for fname in os.listdir(cls_dir):
            path = os.path.join(cls_dir, fname)
            try:
                img = Image.open(path).convert('RGB')
            except Exception as e:
                print(f"⚠️ 파일 로드 실패: {path} => {e}")
                continue
            pred_idx = predictor(img)
            total += 1
            is_correct = (pred_idx == class_names.index(cls))
            if is_correct:
                correct += 1
            print(f"GT: {cls:>6}  -> Pred: {class_names[pred_idx]:>6}  {'✅' if is_correct else '❌'}")

    # 최종 정확도
    acc = correct / total * 100 if total > 0 else 0.0
    print(f"\n✨ Test Accuracy: {acc:.2f}% ({correct}/{total})")

if __name__ == '__main__':
    main()
