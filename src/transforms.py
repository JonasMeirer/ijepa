# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

import torch
import torchvision.transforms as transforms

_GLOBAL_SEED = 0
logger = getLogger()


def make_transforms(
    crop_size=50,  # Changed to match your image size
    crop_scale=(0.3, 1.0),
    horizontal_flip=False,
    normalization=None,  # Will be loaded from dataset_stats.json
    input_is_tensor=False
):
    """
    Create transforms for microscopy dataset.
    
    Args:
        crop_size (int): Size of the crop (50 for your images)
        crop_scale (tuple): Scale range for random resized crop
        horizontal_flip (bool): Whether to apply random horizontal flip
        normalization (tuple): Tuple of (mean, std) for each channel
        input_is_tensor (bool): Whether the input data is already a tensor
    """
    logger.info('making microscopy data transforms')

    transform_list = []
    
    # Random resized crop
    transform_list += [transforms.RandomResizedCrop(
        crop_size,
        scale=crop_scale,
        interpolation=transforms.InterpolationMode.BILINEAR
    )]
    
    # Random horizontal flip
    if horizontal_flip:
        transform_list += [transforms.RandomHorizontalFlip()]
    
    # Convert to tensor and normalize - only if input is not already a tensor
    if not input_is_tensor:
        transform_list += [transforms.ToTensor()]
        
    if normalization is not None:
        transform_list += [transforms.Normalize(normalization[0], normalization[1])]

    transform = transforms.Compose(transform_list)
    return transform
