import torch
import torch.nn as nn
import timm


class FakeImageDetector(nn.Module):
    """
    Generic model wrapper that supports both CNN and ViT architectures
    """

    def __init__(
        self,
        model_name="efficientnet_b0",
        num_classes=2,
        pretrained=True,
        dropout_rate=0.2,
    ):
        super(FakeImageDetector, self).__init__()
        self.model_name = model_name

        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            if len(features.shape) == 4:
                feature_dim = features.shape[1]
                self.global_pool = nn.AdaptiveAvgPool2d(1)
                self.flatten = nn.Flatten()
            else:
                feature_dim = features.shape[-1]
                self.global_pool = None
                self.flatten = None

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)

        if self.global_pool is not None and self.flatten is not None:
            features = self.global_pool(features)
            features = self.flatten(features)
        elif len(features.shape) == 3:
            features = features.mean(dim=1)

        return self.classifier(features)

