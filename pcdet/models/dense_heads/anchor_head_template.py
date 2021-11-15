import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import box_coder_utils, common_utils, loss_utils, box_utils
from .target_assigner.anchor_generator import AnchorGenerator
from .target_assigner.atss_target_assigner import ATSSTargetAssigner
from .target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner
from ...ops.iou3d_nms import iou3d_nms_utils
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def get_center_preds(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:
    """
    x0 = torch.floor(x).long()
    x1 = x0 + 5
    x2 = x0 - 5

    y0 = torch.floor(y).long()
    y1 = y0 + 5
    y2 = y0 - 5

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)
    x2 = torch.clamp(x2, 0, im.shape[1] - 1)
    y2 = torch.clamp(y2, 0, im.shape[0] - 1)
    # print((x0, x1, x2, y0, y1, y2))
    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ie = im[y2, x0]
    Ic = im[y0, x1]
    Id = im[y2, x2]

    ans = torch.cat([Ia, Ib, Ie, Ic, Id], dim=-1)
    return ans

class AnchorHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)
        self.point_cloud_range = point_cloud_range
        self.voxel_size = model_cfg.VOXEL_SIZE

        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(
            num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6),
            **anchor_target_cfg.get('BOX_CODER_CONFIG', {})
        )

        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG
        anchors, self.num_anchors_per_location = self.generate_anchors(
            anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size
        )
        self.anchors = [x.cuda() for x in anchors]
        self.target_assigner = self.get_target_assigner(anchor_target_cfg)

        self.forward_ret_dict = {}
        self.build_losses(self.model_cfg.LOSS_CONFIG)

    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg
        )
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)

        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list

    def get_target_assigner(self, anchor_target_cfg):
        if anchor_target_cfg.NAME == 'ATSS':
            target_assigner = ATSSTargetAssigner(
                topk=anchor_target_cfg.TOPK,
                box_coder=self.box_coder,
                use_multihead=self.use_multihead,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(
                model_cfg=self.model_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        else:
            raise NotImplementedError
        return target_assigner

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        # self.add_module(
        #     'center_loss_func',
        #     loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        # )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes
        )
        return targets_dict

    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        x_idxs = (keypoints[:, :, 0] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, :, 1] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride

        point_bev_features_list = []
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k]
            cur_y_idxs = y_idxs[k]
            cur_bev_features = bev_features[k] # (H, W, C)
            point_bev_features = get_center_preds(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features.unsqueeze(dim=0))

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
        return point_bev_features

    def get_cls_layer_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']
        batch_size = int(cls_preds.shape[0])

        ################################################################
        # center_preds = self.forward_ret_dict['center_preds']
        # centers = self.forward_ret_dict['centers']
        # bev_stride = 2
        # # create 2 kernels
        # m1 = (0, 0)
        # s1 = np.eye(2)*0.1
        # k1 = multivariate_normal(mean=m1, cov=s1)
        # # create a grid of (x,y) coordinates at which to evaluate the kernels
        # xlim = (-0.4, 0.4)
        # ylim = (-0.4, 0.4)
        # xres = 5
        # yres = 5
        # x = np.linspace(xlim[0], xlim[1], xres)
        # y = np.linspace(ylim[0], ylim[1], yres)
        # xx, yy = np.meshgrid(x,y)

        # # evaluate kernels at grid points
        # xxyy = np.c_[xx.ravel(), yy.ravel()]
        # zz = k1.pdf(xxyy)
        # zz_max = np.max(zz)
        # zz_min = np.min(zz)
        # zz_norm = (zz - zz_min)/(zz_max - zz_min)

        # # reshape and plot image
        # img = zz_norm.reshape((xres, yres))
        # # plt.imshow(img); plt.show()
        # center_gt = []
        # center_mask = []
        # for k in range(batch_size):
        #     cur_center = centers[k]
        #     # cur_center_pred = cls_preds[k]
        #     cur_center_gt = torch.zeros(*list(cls_preds[0, :, :, 0].shape), dtype=cls_preds.dtype, device=cls_preds.device)
        #     cur_center_mask = torch.zeros(*list(cls_preds[0, :, :, 0].shape), dtype=torch.long, device=cls_preds.device)
        #     x_idxs = (cur_center[:, 0] - self.point_cloud_range[0]) / self.voxel_size[0]
        #     y_idxs = (cur_center[:, 1] - self.point_cloud_range[1]) / self.voxel_size[1]
        #     x_idxs = x_idxs / bev_stride
        #     y_idxs = y_idxs / bev_stride
        #     x0 = torch.floor(x_idxs).long()
        #     y0 = torch.floor(y_idxs).long()
        #     x1 = x0 + 2
        #     x2 = x0 - 2
        #     y1 = y0 + 2
        #     y2 = y0 - 2
        #     x0 = torch.clamp(x0, 0, cur_center_gt.shape[1] - 1)
        #     x1 = torch.clamp(x1, 0, cur_center_gt.shape[1] - 1)
        #     y0 = torch.clamp(y0, 0, cur_center_gt.shape[0] - 1)
        #     y1 = torch.clamp(y1, 0, cur_center_gt.shape[0] - 1)
        #     x2 = torch.clamp(x2, 0, cur_center_gt.shape[1] - 1)
        #     y2 = torch.clamp(y2, 0, cur_center_gt.shape[0] - 1)
        #     for i in range(len(x1)):
        #         img_temp = torch.from_numpy(img[2-(y0[i]-y2[i]):3+(y1[i]-y0[i]), 2-(x0[i]-x2[i]):3+(x1[i]-x0[i])])
        #         cur_center_gt[y2[i]:y1[i]+1, x2[i]:x1[i]+1] = img_temp
        #         cur_center_mask[y2[i]:y1[i]+1, x2[i]:x1[i]+1] = 1
        #     # plt.imshow(cur_center_gt.cpu().numpy()); plt.show()
        #     center_mask.append(cur_center_mask)
        #     center_gt.append(cur_center_gt)
        # center_gt = torch.stack(center_gt, dim=0)
        # center_mask = torch.stack(center_mask, dim=0)

        # # center_loss = F.binary_cross_entropy(torch.sigmoid(center_preds.squeeze(1)), center_gt.float(), reduction='none')
        # center_loss = self.center_loss_func(center_preds.squeeze(1), center_gt.float(), weights=center_mask)
        # # cls_valid_mask = (rcnn_cls_labels >= 0).float()
        # center_loss = center_loss.sum() / torch.clamp(center_mask.sum(), min=1.0)
        # center_loss = center_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['center_weight']
        #################################################################################################
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        # cls_loss = cls_loss + center_loss
        #############
        # proposal_to_anchor = self.get_proposal_to_gt(box_reg_preds, proposal_gt_boxes, batch_size)
        # score = torch.max(cls_preds, dim=2)[0]
        # score_norm = torch.sigmoid(score)
        
        # score_norm_loss = F.binary_cross_entropy(input=score_norm, target=proposal_to_anchor.detach(), reduction='none')
        # score_norm_loss = (score_norm_loss*cls_weights).sum() / batch_size
        # score_norm_loss = score_norm_loss*self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['score_weight']
        # pillar_cross = (- pillar_features*torch.log2(pillar_features)).view(-1)
        # pillar_cross_num = len(pillar_cross)
        # pillar_cross_loss = sum(pillar_cross) / pillar_cross_num
        # pillar_cross_loss = pillar_cross_loss*self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['pillar_weight']
        # cls_loss = cls_loss + pillar_cross_loss
        tb_dict = {
            'rpn_loss_cls': cls_loss.item(),
            # 'rpn_loss_center':center_loss.item(),
            # 'pillar_cross_loss': pillar_cross_loss.item()
        }
        return cls_loss, tb_dict
    
    @staticmethod
    def get_proposal_to_gt(proposal, gt_boxes, batch_size):
        num_proposals = proposal.size()[1]
        proposal_to_gt = torch.ones((batch_size, num_proposals,), dtype=torch.float, device=proposal.device) * -1
        for k in range(batch_size):
            cur_gt = gt_boxes[k]
            cnt = cur_gt.__len__() - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]
            cur_proposal = proposal[k]
            cur_proposal_to_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(cur_proposal[:, 0:7], cur_gt[:, 0:7])
            proposl_to_gt_argmax = torch.from_numpy(cur_proposal_to_gt_overlap.cpu().detach().numpy().argmax(axis=1)).cuda()
            cur_proposal_to_gt = cur_proposal_to_gt_overlap[
                torch.arange(num_proposals, device=proposal.device), proposl_to_gt_argmax
            ]
            # gt_to_anchor_argmax = torch.from_numpy(cur_proposal_to_gt_overlap.cpu().detach().numpy().argmax(axis=0)).cuda()
            # gt_to_anchor_max = cur_proposal_to_gt_overlap[gt_to_anchor_argmax, torch.arange(cnt+1, device=proposal.device)]           
            proposal_to_gt[k] = cur_proposal_to_gt
        return proposal_to_gt

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])
        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError
