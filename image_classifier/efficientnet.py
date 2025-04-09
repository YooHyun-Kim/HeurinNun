# image_classifier/efficientnet.py
from torchvision.models import efficientnet_b0
import torch.nn as nn

def build_efficientnet(num_classes):
    model = efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
