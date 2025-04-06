# image_classifier/densenet.py
import torchvision.models as models
import torch.nn as nn

def build_densenet(num_classes):
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model
