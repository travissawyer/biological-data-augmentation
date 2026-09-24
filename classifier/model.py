"""VGG16 binary classifier used for the downstream task.

GlobalMaxPool -> Dense(1024, ReLU) -> Dropout -> Dense(1024, ReLU) -> Dropout
-> Dense(1, sigmoid). The backbone is frozen except the convolutions among the
last N Keras-equivalent VGG16 layers, and dropout is inserted after each of
those layers. An explicit L2 penalty is applied to the trainable convolution
weights and to all three head weight matrices.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights

# Keras VGG16(include_top=False) layer order mapped onto torchvision module indices,
# so that "the last N layers" means the same thing in both frameworks.
VGG16_KERAS_LAYER_MAP = [
    {"name": "block1_conv1", "type": "conv", "conv_module_idx": 0,  "end_module_idx": 1},
    {"name": "block1_conv2", "type": "conv", "conv_module_idx": 2,  "end_module_idx": 3},
    {"name": "block1_pool",  "type": "pool", "conv_module_idx": None, "end_module_idx": 4},
    {"name": "block2_conv1", "type": "conv", "conv_module_idx": 5,  "end_module_idx": 6},
    {"name": "block2_conv2", "type": "conv", "conv_module_idx": 7,  "end_module_idx": 8},
    {"name": "block2_pool",  "type": "pool", "conv_module_idx": None, "end_module_idx": 9},
    {"name": "block3_conv1", "type": "conv", "conv_module_idx": 10, "end_module_idx": 11},
    {"name": "block3_conv2", "type": "conv", "conv_module_idx": 12, "end_module_idx": 13},
    {"name": "block3_conv3", "type": "conv", "conv_module_idx": 14, "end_module_idx": 15},
    {"name": "block3_pool",  "type": "pool", "conv_module_idx": None, "end_module_idx": 16},
    {"name": "block4_conv1", "type": "conv", "conv_module_idx": 17, "end_module_idx": 18},
    {"name": "block4_conv2", "type": "conv", "conv_module_idx": 19, "end_module_idx": 20},
    {"name": "block4_conv3", "type": "conv", "conv_module_idx": 21, "end_module_idx": 22},
    {"name": "block4_pool",  "type": "pool", "conv_module_idx": None, "end_module_idx": 23},
    {"name": "block5_conv1", "type": "conv", "conv_module_idx": 24, "end_module_idx": 25},
    {"name": "block5_conv2", "type": "conv", "conv_module_idx": 26, "end_module_idx": 27},
    {"name": "block5_conv3", "type": "conv", "conv_module_idx": 28, "end_module_idx": 29},
    {"name": "block5_pool",  "type": "pool", "conv_module_idx": None, "end_module_idx": 30},
]


class VGG16BinaryClassifier(nn.Module):
    def __init__(self, dropout_p: float = 0.1, num_train_keras_layers: int = 10,
                 pretrained: bool = True):
        super().__init__()
        if not 1 <= num_train_keras_layers <= len(VGG16_KERAS_LAYER_MAP):
            raise ValueError(f"num_train_keras_layers must be 1..{len(VGG16_KERAS_LAYER_MAP)}")

        base = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        feature_layers = list(base.features.children())
        last = VGG16_KERAS_LAYER_MAP[-num_train_keras_layers:]
        self.unfreeze_conv_indices = [l["conv_module_idx"] for l in last if l["type"] == "conv"]
        self.dropout_after_indices = [l["end_module_idx"] for l in last]

        if pretrained:
            for p in base.features.parameters():
                p.requires_grad = False
            for idx in self.unfreeze_conv_indices:
                for p in feature_layers[idx].parameters():
                    p.requires_grad = True
        else:
            # Nothing pretrained to preserve, so the whole backbone trains. The L2
            # scope and dropout placement are left unchanged, so this differs from
            # the pretrained setting in exactly one respect.
            #
            # Note: the penalty that suits a pretrained backbone is often far too
            # strong from a random start. A pretrained network has informative
            # features to lock onto while its weights shrink; from random init
            # there is no such signal, and training can collapse to a constant
            # prediction. A fair from-scratch control usually needs a much smaller
            # --reg_strength.
            for p in base.features.parameters():
                p.requires_grad = True

        modified = []
        for idx, layer in enumerate(feature_layers):
            modified.append(layer)
            if idx in self.dropout_after_indices:
                modified.append(nn.Dropout(p=dropout_p))
        self.features = nn.Sequential(*modified)

        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_p)
        self.sigmoid = nn.Sigmoid()

        for layer in (self.fc1, self.fc2, self.fc3):    # Keras glorot_uniform
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        self._regularized = [feature_layers[i].weight for i in self.unfreeze_conv_indices]
        self._regularized += [self.fc1.weight, self.fc2.weight, self.fc3.weight]

    def l2_penalty(self, reg_strength: float) -> torch.Tensor:
        penalty = torch.zeros(1, device=self.fc1.weight.device)
        for p in self._regularized:
            penalty = penalty + torch.sum(p.pow(2))
        return reg_strength * penalty

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(self.global_max_pool(x), 1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.sigmoid(self.fc3(x))
