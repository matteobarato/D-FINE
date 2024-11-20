"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.utils.data

import torchvision

from PIL import Image
import faster_coco_eval
import faster_coco_eval.core.mask as coco_mask
from ._dataset import DetDataset
from .coco_dataset import CocoDetection
from .._misc import convert_to_tv_tensor
from ...core import register

torchvision.disable_beta_transforms_warning()
faster_coco_eval.init_as_pycocotools()
Image.MAX_IMAGE_PIXELS = None

__all__ = ['ConcatCocoDetection']


@register()
class ConcatCocoDetection(torch.utils.data.dataset.ConcatDataset, DetDataset):
    __inject__ = ['transforms', ]
    __share__ = ['remap_mscoco_category']
        
    def __init__(self, datasets, transforms, return_masks=False, remap_mscoco_category=False):        
        self.datasets = [
            CocoDetection(img_folder=dataset['img_folder'], ann_file=dataset['ann_file'], transforms=transforms, return_masks=return_masks, remap_mscoco_category=remap_mscoco_category)
            for dataset in datasets
        ]
        self._epoch = 0
        super(ConcatCocoDetection, self).__init__(self.datasets)
        
    def set_epoch(self, epoch) -> None:
        self._epoch = epoch
        for d in self.datasets:
            d.set_epoch(epoch)