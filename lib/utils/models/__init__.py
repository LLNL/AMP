from .resnet import *
from .wideresnet import *
from .resnetv2 import *
from .wrn import *

MODEL_DICT = {}
MODEL_DICT['resnet34'] = ('ResNet34',ResNet34)
MODEL_DICT['resnet18'] = ('ResNet18',ResNet18)
MODEL_DICT['resnet20'] = ('ResNet20',resnet20)
MODEL_DICT['wideresnet'] = ('WideResNet',WideResNet)
MODEL_DICT['widenresnet28'] = ('WideResNet28',WideResNet28)
