from .partA2_head import PartA2FCHead
from .pointrcnn_head import PointRCNNHead
from .pvrcnn_head import PVRCNNHead
from .rcnet_head import RCHead
from .roi_head_template import RoIHeadTemplate
from .ascnet_head import ASCNetHead

__all__ = {
    'RoIHeadTemplate': RoIHeadTemplate,
    'PartA2FCHead': PartA2FCHead,
    'PVRCNNHead': PVRCNNHead,
    'PointRCNNHead': PointRCNNHead,
    'RCHead': RCHead,
    'ASCNetHead': ASCNetHead
}
