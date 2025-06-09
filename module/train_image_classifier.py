import os
import sys
import argparse
from collections import Counter

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 프로젝트 루트를 PYTHONPATH에 포함
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from image_classifier.resnet import build_resnet
from image_classifier.densenet import build_densenet


def train_model(model: nn.Module,
                dataloaders: dict[str, DataLoader],
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device,
                num_epochs: int) -> nn.Module:
    model.to(device)
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f"[Epoch {epoch}/{num_epochs}]")
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                total_samples += inputs.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects / total_samples
            print(f"  {phase} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print(f"Best val Acc: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model


def main():
    parser = argparse.ArgumentParser(description="이미지 분류기 학습 (ResNet / DenseNet)")
    parser.add_argument('--model', required=True, choices=['resnet', 'densenet'],
                        help='백본 모델 선택')
    parser.add_argument('--data_dir', required=True,
                        help='train/val/test 폴더가 있는 상위 디렉터리')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='배치 크기')
    parser.add_argument('--epochs', type=int, default=10,
                        help='학습 epoch 수')
    args = parser.parse_args()

    # 데이터 경로
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir   = os.path.join(args.data_dir, 'val')
    test_dir  = os.path.join(args.data_dir, 'test')

    # 클래스 세팅
    train_ds = datasets.ImageFolder(train_dir)
    classes = train_ds.classes
    num_classes = len(classes)
    labels = [label for _, label in train_ds.samples]
    print(f"Detected classes: {classes}")
    print(f"Label distribution: {Counter(labels)}")

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # DataLoader
    dataloaders = {
        'train': DataLoader(datasets.ImageFolder(train_dir, transform), batch_size=args.batch_size, shuffle=True),
        'val':   DataLoader(datasets.ImageFolder(val_dir,   transform), batch_size=args.batch_size)
    }

    # 모델 초기화
    if args.model == 'resnet':
        model = build_resnet(num_classes=num_classes, feature_only=False)
    else:  # densenet
        model = build_densenet(num_classes=num_classes, feature_only=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 학습
    model = train_model(model, dataloaders, criterion, optimizer, device,
                        num_epochs=args.epochs)

    # 테스트 정확도 출력
    test_ds = datasets.ImageFolder(test_dir, transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
    print(f"Test Acc: {correct/total:.4f}")

    # **학습된 모델 가중치 저장**
    save_dir = os.path.join(PROJECT_ROOT, 'module')
    os.makedirs(save_dir, exist_ok=True)
    ckpt_name = 'resnet_classifier.pth' if args.model == 'resnet' else 'densenet_classifier.pth'
    ckpt_path = os.path.join(save_dir, ckpt_name)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model weights saved to {ckpt_path}")


if __name__ == '__main__':
    main()
