# evaluate_single_image.py
import argparse
from PIL import Image
import torch

def main():
    parser = argparse.ArgumentParser(
        description="단일 이미지 분류 (ResNet/DenseNet/Tip-Adapter)"
    )
    parser.add_argument(
        '--model',
        choices=['resnet','densenet','tip_adapter'],
        required=True,
        help='사용할 백본 모델'
    )
    parser.add_argument(
        'image_path',
        help='분류할 이미지 파일 경로'
    )
    # ResNet/DenseNet 체크포인트는 필요 시 수정
    parser.add_argument(
        '--resnet_ckpt',
        default='module/resnet_classifier.pth',
        help='ResNet 체크포인트 경로'
    )
    parser.add_argument(
        '--densenet_ckpt',
        default='module/densenet_classifier.pth',
        help='DenseNet 체크포인트 경로'
    )
    args = parser.parse_args()

    img = Image.open(args.image_path).convert("RGB")

    if args.model in ['resnet','densenet']:
        import torch.nn.functional as F
        # build & load 모델
        if args.model == 'resnet':
            from image_classifier.resnet import build_resnet, preprocess_image
            model = build_resnet(feature_only=False)
            ckpt = args.resnet_ckpt
        else:
            from image_classifier.densenet import build_densenet, preprocess_image
            model = build_densenet(feature_only=False)
            ckpt = args.densenet_ckpt

        # checkpoint 로드
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        # 전처리 & 예측
        img_t = preprocess_image(img)
        with torch.no_grad():
            logits = model(img_t)
            pred = logits.argmax(dim=-1).item()
        # 클래스 이름 매핑
        from image_classifier.resnet import class_names as resnet_names  # 두 모듈 동일하게 정의했다고 가정
        class_names = resnet_names if args.model=='resnet' else resnet_names  # 동일 리스트 사용
        print(f"[{args.model}] {args.image_path} → {class_names[pred]}")

    else:  # tip_adapter
        from tip_adapter.tip_adapter import predict_image_tip_adapter
        pred = predict_image_tip_adapter(img)  # 예: ['흐름도'], [] 등
        # '기타' 이 반환될 경우 빈 리스트이니 처리
        if not pred:
            print(f"[tip_adapter] {args.image_path} → 기타로 분류되어 출력하지 않음")
        else:
            print(f"[tip_adapter] {args.image_path} → {pred[0]}")

if __name__ == "__main__":
    main()
