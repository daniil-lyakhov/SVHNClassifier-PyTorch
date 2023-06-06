from typing import Any, Callable, List, Optional
from torch import nn
from torchvision.models import MobileNetV3
from torchvision.models.mobilenetv3 import InvertedResidualConfig
from torchvision.models.mobilenetv3 import _mobilenet_v3_conf

from modeling.base_model import BaseModelMixIn
from modeling.base_model import ClassificationHead


class SVGNMobielentV3(MobileNetV3, BaseModelMixIn):
    def __init__(self, inverted_residual_setting: List[InvertedResidualConfig], last_channel: int, num_classes: int = 1000, block = None, norm_layer = None, dropout: float = 0.2, **kwargs: Any) -> None:
        super().__init__(inverted_residual_setting, last_channel, num_classes, block, norm_layer, dropout, **kwargs)
        in_features = self.classifier[0].in_features
        self.classifier = ClassificationHead(in_features)


def get_mobilenet_v3_small(progress: bool = True, **kwargs):
    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small", **kwargs)
    model = SVGNMobielentV3(inverted_residual_setting, last_channel, **kwargs)
    return model


def get_mobilenet_v3_large(progress: bool = True, **kwargs):
    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_large", **kwargs)
    model = SVGNMobielentV3(inverted_residual_setting, last_channel, **kwargs)
    return model
