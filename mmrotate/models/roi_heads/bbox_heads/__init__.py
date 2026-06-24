# Copyright (c) OpenMMLab. All rights reserved.
from .convfc_rbbox_head import (RotatedConvFCBBoxHead,
                                RotatedKFIoUShared2FCBBoxHead,
                                RotatedShared2FCBBoxHead)
from .gv_bbox_head import GVBBoxHead
from .rotated_bbox_head import RotatedBBoxHead
from .hap_head import HAPHead
from .newHead3 import ARCRotatedBBoxHead
from .strip_head import StripHead_,StripHead
from .ARHead import ARConvRegBBoxHead
__all__ = [
    'RotatedBBoxHead', 'RotatedConvFCBBoxHead', 'RotatedShared2FCBBoxHead',
    'GVBBoxHead', 'RotatedKFIoUShared2FCBBoxHead', 'HAPHead' ,  'ARCRotatedBBoxHead', 'StripHead_', 'StripHead', 'ARConvRegBBoxHead'
]
