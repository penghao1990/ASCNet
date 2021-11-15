import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
# import torch.utils.data.sampler as sp

from collections import OrderedDict

from .vfe_template import VFETemplate
from ....ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ....ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from ....utils import common_utils


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        out_channels = out_channels
        self.hidden_size = 128
        self.attention_channel = 4

        self.part = 50000
        # 非线性编码
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=1,  bias=False),
            nn.BatchNorm1d(64, eps=1e-5, momentum=0.01),
            nn.ReLU()
        )

        self.rnn_vfe = nn.GRU(
            input_size=64, hidden_size=128, num_layers=2, batch_first=True, 
            bidirectional=False
        )

    def forward(self, inputs, voxel_num_points, mask):
        max_num, max_num_index = torch.max(voxel_num_points, dim=0)
        if max_num < 16:
            inputs = inputs[:, :max_num.int(), :]
        x = inputs.permute(0, 2, 1)

        encoder_feature = self.encoder(x).permute(0, 2, 1)
        max_encoder_feature = torch.max(encoder_feature, dim=1)[0]

        x_pack = rnn_utils.pack_padded_sequence(
            encoder_feature, voxel_num_points.cpu(), batch_first=True, enforce_sorted=False
        )
        batch_size = list(encoder_feature.size())[0]
        h0 = self.initHidden(batch_size)
        h0 = h0.to(encoder_feature.device)
        h0.requires_grad_(True)
        x_rnn, _ = self.rnn_vfe(x_pack, h0)
        x_rnn_pad, rnn_voxel_num_points = rnn_utils.pad_packed_sequence(
            x_rnn, batch_first=True, padding_value=0
        )

        x_rnn_index = rnn_voxel_num_points.numpy() - 1
        idx = np.arange(0, batch_size)
        x_rnn_end = x_rnn_pad[idx, x_rnn_index, :]
        x_out = torch.cat([max_encoder_feature, x_rnn_end], dim=-1)

        if self.last_vfe:
            return x_out
        else:
            return x

    def initHidden(self, batch_size):
        h0 = torch.zeros(2, batch_size, self.hidden_size)
        return h0


class PSCFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        vfe_mlps = self.model_cfg['POINT_FE'].MLPS
        for k in range(len(vfe_mlps)):
            vfe_mlps[k] = [1] + vfe_mlps[k]
        self.SA_vfe = pointnet2_stack_modules.StackSAModuleMSG(
            radii=self.model_cfg['POINT_FE'].POOL_RADIUS,
            nsamples=self.model_cfg['POINT_FE'].NSAMPLE,
            mlps=vfe_mlps,
            use_xyz=True,
            pool_method='max_pool'
        )   

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_sampled_points(self, batch_dict):
        batch_size = batch_dict['batch_size']
        if self.model_cfg.POINT_SOURCE == 'raw_points':
            src_points = batch_dict['points'][:, 1:4]
            batch_indices = batch_dict['points'][:, 0].long()
        elif self.model_cfg.POINT_SOURCE == 'voxel_centers':
            src_points = common_utils.get_voxel_centers(
                batch_dict['voxel_coords'][:, 1:4],
                downsample_times=1,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            batch_indices = batch_dict['voxel_coords'][:, 0].long()
        else:
            raise NotImplementedError
        keypoints_list = []
        for bs_idx in range(batch_size):
            bs_mask = (batch_indices == bs_idx)
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)  # (1, N, 3)
            if self.model_cfg.SAMPLE_METHOD == 'FPS':
                cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(
                    sampled_points[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS
                ).long()

                if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                    empty_num = self.model_cfg.NUM_KEYPOINTS - sampled_points.shape[1]
                    cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]

                keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)

            elif self.model_cfg.SAMPLE_METHOD == 'FastFPS':
                raise NotImplementedError
            else:
                raise NotImplementedError

            keypoints_list.append(keypoints)

        keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3)
        return keypoints

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):

        points_ori = batch_dict['points']
        batch_size = batch_dict['batch_size']
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        xyz = points_ori[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = points_ori[:, 0]
        vfe_xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        vfe_batch_idx = coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()
            vfe_xyz_batch_cnt[k] = (vfe_batch_idx == k).sum()

        points_ori_features = points_ori[:, 4].view(-1, 1)

        vfe_xyz = points_mean.view(-1, 3)
        vfe_points, vfe_points_features = self.SA_vfe(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=vfe_xyz.contiguous(),
            new_xyz_batch_cnt=vfe_xyz_batch_cnt,
            features=points_ori_features.contiguous(),
        )

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:]]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask_bool = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask_bool, -1).type_as(voxel_features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features, voxel_num_points, mask_bool)
        
        batch_dict['sa_points'] = xyz
        batch_dict['sa_batch_cnt'] = xyz_batch_cnt
        batch_dict['sa_features'] = points_ori_features
        features = torch.cat([features, vfe_points_features], dim=1)
        batch_dict['pillar_features'] = features
        return batch_dict
