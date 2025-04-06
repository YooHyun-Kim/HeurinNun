# image_classifier/resnet.py
import torchvision.models as models
import torch.nn as nn

def build_resnet(num_classes):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
