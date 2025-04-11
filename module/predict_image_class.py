import torch
import os
from PIL import Image  # PIL 모듈 추가
from image_classifier.resnet import preprocess_image, get_image_features
from image_classifier.resnet import build_resnet

# 클래스명 리스트
class_names = ["흐름도", "건축도면", "디바이스", "장비도면", "회로도면"]

# 모델 로드 (이미 학습된 모델을 로드)
model = build_resnet(feature_only=False)  # feature_only=False로 설정하면 마지막 FC 레이어 포함
model_path = os.path.join(os.path.dirname(__file__), "resnet_classifier.pth")


def predict_image(image_path):
    # 이미지 로드 및 전처리
    img_pil = Image.open(image_path)  # PIL을 사용하여 이미지 열기
    img_tensor = preprocess_image(img_pil)  # 이미지 전처리

    # 모델 예측
    model.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():  # 파라미터 업데이트를 하지 않음
        output = model(img_tensor)  # 모델 예측
        _, predicted = output.max(1)  # 예측된 클래스의 인덱스
        predicted_class = class_names[predicted.item()]  # 클래스명으로 변환
    
    return predicted_class

def evaluate_test_folder(test_root):
    total = 0
    correct = 0
    
    if os.path.isdir(test_root):  # 폴더가 있을 때
        for class_name in class_names:
            class_path = os.path.join(test_root, class_name)
            
            if os.path.isdir(class_path):  # 클래스 폴더가 있을 때
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    pred = predict_image(img_path)
                    total += 1
                    if pred == class_name:
                        correct += 1
                    print(f"[✓] GT: {class_name}, Pred: {pred}, {'✅' if pred == class_name else '❌'}")
    elif os.path.isfile(test_root):  # 파일 경로일 때
        pred = predict_image(test_root)
        total += 1
        if pred == "기대하는 클래스":  # 기대하는 클래스를 정의
            correct += 1
        print(f"[✓] Pred: {pred}, {'✅' if pred == '기대하는 클래스' else '❌'}")
    
    # 정확도 계산
    if total > 0:
        acc = correct / total * 100
        print(f"\n🎯 전체 정확도: {acc:.2f}% ({correct}/{total})")
    else:
        print("테스트할 이미지가 없습니다.")

if __name__ == "__main__":
    evaluate_test_folder("data/page_3.png")
