import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FeatureExtractor(nn.Module):
    """ResNet18 backbone for mapping raw images to feature vectors."""
    def __init__(self, feature_dim=512, num_classes=2):
        super().__init__()
        resnet = models.resnet18(weights='DEFAULT')
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc_features = nn.Linear(512, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):
        features = self.backbone(x).view(x.size(0), -1)
        features = F.relu(self.fc_features(features))
        logits = self.classifier(features)
        return features, logits