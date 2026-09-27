from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import json
from datetime import datetime
from pathlib import Path
from torchvision.models import resnet50, ResNet50_Weights


class Backbone(nn.Module, ABC):
    """
    Abstract base for all 4 baseline backbones (NonTemporal/Temporal x frame/crop).

    Defines the shared checkpoint-saving contract: each concrete subclass implements
    save() to persist its own components (resnet, lstm, etc.) via _save_component,
    keyed by backbone_name under save_dir. Loading is handled separately via the
    static load_model(), which loads weights into any already-constructed nn.Module --
    not tied to reconstructing a specific Backbone instance, since downstream usage
    mixes components across backbones.
    """

    @abstractmethod
    def forward(self, x):
        ...

    def _save_component(self, save_dir: str, module: nn.Module, backbone_name: str,
                         component: str, meta: dict) -> str:
        """Save one submodule's state_dict + a JSON metadata sidecar, timestamped for versioning."""
        backbone_dir = Path(save_dir) / backbone_name
        backbone_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        pt_path = backbone_dir / f"{component}_{timestamp}.pt"
        json_path = backbone_dir / f"{component}_{timestamp}.json"

        torch.save(module.state_dict(), pt_path)
        meta = {**meta, "backbone_name": backbone_name, "component": component, "timestamp": timestamp}

        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        return str(pt_path)

    @abstractmethod
    def save(self, save_dir: str, backbone_name: str, meta: dict):
        ...

    @staticmethod
    def load_model(module: nn.Module, save_dir: str, backbone_name: str,
                    component: str, tag: str = "best", map_location='cpu') -> dict:
        """Load a saved component's weights into `module` in-place; returns its saved metadata dict."""

        backbone_dir = Path(save_dir) / backbone_name
        weights = torch.load(backbone_dir / f"{component}_{tag}.pt", map_location=map_location)
        module.load_state_dict(weights, strict=True)

        with open(backbone_dir / f"{component}_{tag}.json") as f:
            return json.load(f)


class NonTemporalBackbone(Backbone):
    """
    frame/crop -> resnet50 (feature extractor) -> linear classifier. No temporal modeling.

    Used for Backbone-1 (frames, 8 group-activity classes, image_level=True) and
    Backbone-2 (crops, 9 individual-action classes, image_level=False) -- same
    architecture, different num_classes and input type. Input type (frame vs. crop)
    is a Dataset-level distinction, not encoded here, since the model itself is
    agnostic to what kind of image it receives.
    """
    
    def __init__(self, image_level=True):
        super().__init__()
        self.num_classes = 8 if image_level else 9

        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.feat_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.feature_extractor = resnet

        self.classifier = nn.Linear(in_features=self.feat_dim, out_features=self.num_classes)

    def forward(self, x):
        feats = self.feature_extractor(x)
        return self.classifier(feats)

    def save(self, save_dir: str, backbone_name: str, meta: dict):
        return self._save_component(
            save_dir=save_dir, module=self.feature_extractor,
            backbone_name=backbone_name, component="resnet-50", meta=meta,
        )


class TemporalBackbone(Backbone):
    """
    clip -> resnet50 (feature extractor, shared across frames) -> LSTM (temporal) -> linear classifier.

    Used for Backbone-3 (frame clips, 8 group-activity classes, image_level=True) and
    Backbone-4 (crop clips, 9 individual-action classes, image_level=False) -- same
    architecture, different num_classes and input type. The resnet processes each
    frame in the clip independently (frames folded into the batch dim), and the LSTM
    models how that per-frame representation evolves across time.
    """

    def __init__(self, image_level=True, hidden_size=512):
        super().__init__()
        self.image_level = image_level
        self.num_classes = 8 if image_level else 9

        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.feat_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.feature_extractor = resnet

        self.lstm = nn.LSTM(input_size=self.feat_dim, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(in_features=hidden_size, out_features=self.num_classes)

    def forward(self, x):
        if self.image_level:
            B, T, CH, H, W = x.shape
            feats = self.feature_extractor(x.view(B * T, CH, H, W))
            feats = feats.view(B, T, -1)
            _, (h_st, _) = self.lstm(feats)
            return self.classifier(h_st[-1])          # (B, num_classes)
        else:
            B, T, P, CH, H, W = x.shape
            feats = self.feature_extractor(x.view(B * T * P, CH, H, W))
            feats = feats.view(B, T, P, -1).permute(0, 2, 1, 3).reshape(B * P, T, -1)   # (B*P, T, feat_dim)

            _, (h_st, _) = self.lstm(feats)
            out = self.classifier(h_st[-1])          # (B*P, num_classes)
            return out.view(B, P, -1)                  # (B, P, num_classes)


    def save(self, save_dir: str, backbone_name: str, meta: dict):
        resnet_path = self._save_component(
            save_dir=save_dir, module=self.feature_extractor,
            backbone_name=backbone_name, component="resnet-50", meta=meta,
        )
        lstm_path = self._save_component(
            save_dir=save_dir, module=self.lstm,
            backbone_name=backbone_name, component="lstm", meta=meta,
        )
        return resnet_path, lstm_path