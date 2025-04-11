import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from image_classifier.resnet import build_resnet
from image_classifier.densenet import build_densenet
from image_classifier.efficientnet import build_efficientnet


def get_model(name, num_classes):
    if name == "resnet":
        return build_resnet(num_classes=num_classes)
    elif name == "densenet":
        return build_densenet(num_classes=num_classes)
    elif name == "efficientnet":
        return build_efficientnet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=10):
    model.to(device)

    for epoch in range(num_epochs):
        print(f"\n[Epoch {epoch+1}/{num_epochs}]")
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase.upper()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

    return model


def main():
    data_dir = "data/image_dataset"  # 이 경로는 사용자에 따라 바꿔줘야 함
    model_name = "resnet"  # 또는 "densenet", "efficientnet"
    num_classes = 5
    batch_size = 16
    num_epochs = 10
    lr = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), transform=transform)
        for x in ['train', 'val']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=2)
        for x in ['train', 'val']
    }

    model = get_model(model_name, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model = train_model(model, dataloaders, criterion, optimizer, device, num_epochs=num_epochs)

    torch.save(model.state_dict(), f"{model_name}_classifier.pth")
    print(f"\n✅ 모델 저장 완료: {model_name}_classifier.pth")


if __name__ == "__main__":
    main()
