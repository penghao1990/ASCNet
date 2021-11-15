import torch.nn as nn
import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate


def get_roi_feature(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:
    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans


class ASCNetHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, voxel_size, point_cloud_range, num_class=1):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        mlps = self.model_cfg.ROI_GRID_POOL.MLPS
        for k in range(len(mlps)):
            mlps[k] = [1] + mlps[k]

        self.roi_grid_pool_layer = pointnet2_stack_modules.StackSAModuleMSG(
            radii=self.model_cfg.ROI_GRID_POOL.POOL_RADIUS,
            nsamples=self.model_cfg.ROI_GRID_POOL.NSAMPLE,
            mlps=mlps,
            use_xyz=True,
            pool_method=self.model_cfg.ROI_GRID_POOL.POOL_METHOD,
        )

        self.GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE

        pre_channel = 896
        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),   
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k], eps=1e-5, momentum=0.01),
                nn.ReLU()                 
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]
            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                # shared_fc_list.append(nn.BatchNorm1d(self.model_cfg.SHARED_FC[k], eps=1e-3, momentum=0.01))
                # shared_fc_list.append(nn.ReLU())
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        pre_channel_p = 2432
        shared_fc_list_p = []
        for k in range(0, self.model_cfg.P_FC.__len__()):
            shared_fc_list_p.extend([
                nn.Conv1d(pre_channel_p, self.model_cfg.P_FC[k], kernel_size=1, bias=False), 
                nn.BatchNorm1d(self.model_cfg.P_FC[k], eps=1e-5, momentum=0.01),
                nn.ReLU()                   
            ])
            pre_channel_p = self.model_cfg.P_FC[k]
            if k != self.model_cfg.P_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
            #     shared_fc_list_p.append(nn.BatchNorm1d(self.model_cfg.P_FC[k], eps=1e-3, momentum=0.01))
            #     shared_fc_list_p.append(nn.ReLU())
                shared_fc_list_p.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_p_layer = nn.Sequential(*shared_fc_list_p)

        self.hidden_size_cls = 128
        self.cls_layers = nn.GRU(
            input_size=64, hidden_size=self.hidden_size_cls, num_layers=2, batch_first=True,
            bidirectional=False
        )

        self.cls_linear_up = nn.Sequential(
            nn.Linear(in_features=256, out_features=1, bias=False),
        )
        self.reg_linear_up = nn.Sequential( 
            nn.Linear(in_features=256, out_features=7, bias=False),
        )
        self.cls_linear_mid = nn.Sequential(
            nn.Linear(in_features=256, out_features=1, bias=False),
        )
        self.reg_linear_mid = nn.Sequential(
            nn.Linear(in_features=256, out_features=7, bias=False),
        )
        self.cls_linear_down = nn.Sequential(
            nn.Linear(in_features=256, out_features=1, bias=False),
        )
        self.reg_linear_down = nn.Sequential( 
            nn.Linear(in_features=256, out_features=7, bias=False),
        )
        
        self.init_weights(weight_init='xavier')

    def initHiddenCls(self, batch_size):
        h0 = torch.zeros(2, batch_size, self.hidden_size_cls)
        return h0

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_linear_up[-1].weight, mean=0, std=0.001)
        nn.init.normal_(self.reg_linear_mid[-1].weight, mean=0, std=0.001)
        nn.init.normal_(self.reg_linear_down[-1].weight, mean=0, std=0.001)

    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        x_idxs = (keypoints[:, :, 0] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, :, 1] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride

        point_bev_features_list = []
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k]
            cur_y_idxs = y_idxs[k]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = get_roi_feature(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features.unsqueeze(dim=0))

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
        return point_bev_features

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        roi_labels = batch_dict['roi_labels']
        sa_points = batch_dict['sa_points']
        sa_points_feature = batch_dict['sa_features']
        sa_batch_cnt = batch_dict['sa_batch_cnt']
        global_roi_grid_points, local_roi_grid_points, num_points, max_roi_size = self.get_global_grid_points_of_roi(
            roi_labels, rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        global_points = global_roi_grid_points.contiguous().view(batch_size, -1, 3)
        global_inner_cornerbev_points = global_roi_grid_points[:, :-19, :].contiguous().view(batch_size, -1, 3)

        new_xyz = global_points.view(-1, 3)
        new_xyz_batch_cnt = sa_points.new_zeros(batch_size).int().fill_(global_points.shape[1])
        pooled_points, pooled_points_features = self.roi_grid_pool_layer(
            xyz=sa_points.contiguous(),
            xyz_batch_cnt=sa_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=sa_points_feature.contiguous(),
        )
        pooled_points_features = pooled_points_features.view(
            batch_size, -1,
            pooled_points_features.shape[-1]
        )
        pooled_spatial_features_x = self.interpolate_from_bev_features(
            global_inner_cornerbev_points, batch_dict['spatial_features_2d'], batch_dict['batch_size'],
            bev_stride=2
        )
        return pooled_points_features, pooled_spatial_features_x, num_points, max_roi_size

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
            - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def get_global_grid_points_of_roi(self, roi_labels, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points, num_points, max_roi_size = self.get_dense_grid_points_adaptive(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points, num_points, max_roi_size

    @staticmethod
    def get_dense_grid_points_adaptive(rois, batch_size_rcnn, grid_size):

        corners_center = torch.from_numpy(np.array([[0.5, 0.5, 0.0],
                                                    [0.5, 0.5, 0.5],
                                                    [0.0, 0.5, 0.5],
                                                    [0.5, 0.0, 0.5],
                                                    [0.5, 1.0, 0.5],
                                                    [1.0, 0.5, 0.5],
                                                    [0.5, 0.5, 1.0],
                                                    [0.0, 0.0, 0.0],
                                                    [1.0, 0.0, 0.0],
                                                    [0.0, 1.0, 0.0],
                                                    [1.0, 1.0, 0.0],
                                                    [0.0, 0.0, 0.5],
                                                    [1.0, 0.0, 0.5],
                                                    [0.0, 1.0, 0.5],
                                                    [1.0, 1.0, 0.5],
                                                    [0.0, 0.0, 1.0],
                                                    [1.0, 0.0, 1.0],
                                                    [0.0, 1.0, 1.0],
                                                    [1.0, 1.0, 1.0]], dtype=np.float32)).to(rois.device)

        roi_length_width = rois[:, 3:5]
        roi_length_width_point = torch.ceil(roi_length_width/grid_size).int()
        roi_length_width_point = torch.clamp(roi_length_width_point, min=1)

        num_points = roi_length_width_point[:, 0]*roi_length_width_point[:, 1]
        point_num_unique = torch.unique(roi_length_width_point, dim=0) 
        max_roi_size_vec = point_num_unique[:, 0]*point_num_unique[:, 1]
        max_roi_size = torch.max(max_roi_size_vec, dim=0)[0]   
        roi_grid_points = rois.new_zeros((batch_size_rcnn, max_roi_size+19, 3))

        for i, value in enumerate(point_num_unique):
            num_long = value[0]
            num_width = value[1]
            cur_mask = ((roi_length_width_point[:, 0] == num_long) & (roi_length_width_point[:, 1] == num_width))          
            cur_rois = rois[cur_mask]
            cur_batch_size_rcnn = len(cur_rois)
            faked_features = cur_rois.new_ones((num_long, num_width, 1))
            dense_idx = faked_features.nonzero()+0.5
            grid_size_3d = torch.from_numpy(np.array([num_long, num_width, 1], dtype=np.float32)).to(rois.device)
            dense_idx = (dense_idx) / grid_size_3d
            dense_idx = torch.cat([dense_idx, corners_center], dim=0)
            dense_idx = dense_idx.repeat(cur_batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)
            cur_local_roi_size = cur_rois.view(cur_batch_size_rcnn, -1)[:, 3:6]
            cur_roi_grid_points = (dense_idx) * cur_local_roi_size.unsqueeze(dim=1) \
                - (cur_local_roi_size.unsqueeze(dim=1) / 2)
            roi_grid_points[cur_mask, :(num_long*num_width + 19), :] = cur_roi_grid_points

        return roi_grid_points, num_points, max_roi_size

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
       
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        up_mask = batch_dict['roi_labels'].view(-1)==1
        mid_mask = batch_dict['roi_labels'].view(-1)==3
        down_mask =batch_dict['roi_labels'].view(-1)==2

        # RoI aware pooling
        pooled_points_features, pooled_features, num_points, max_roi_size= self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)
        pooled_features = pooled_features.view(
            -1, max_roi_size,
            pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)
        pooled_points_features = pooled_points_features.view(
            -1, max_roi_size+19, 
            pooled_points_features.shape[-1]
        ) 
        batch_size_rcnn = pooled_features.shape[0]

        pooled_features = torch.cat([pooled_features, pooled_points_features[:,:-19,:]], dim=-1)
        pooled_points_features = pooled_points_features[:,-19:,:]
        pooled_points_features = pooled_points_features.view(batch_size_rcnn, -1, 1).contiguous() 
        pooled_features = pooled_features.permute(0, 2, 1).contiguous() 
        cls_features = self.shared_fc_layer(pooled_features)
        reg_features = self.shared_fc_p_layer(pooled_points_features).squeeze(-1)
       
        cls_features = cls_features.permute(0, 2, 1)
        max_cls_feature = torch.max(cls_features, dim=1)[0]

        shared_features_pack = rnn_utils.pack_padded_sequence(
            cls_features, num_points.cpu(), batch_first=True, enforce_sorted=False
        )
        hcls_weight = self.initHiddenCls(batch_size_rcnn)
        hcls_weight = hcls_weight.to(cls_features.device)
        hcls_weight.requires_grad_(True)
        cls_rnn, _ = self.cls_layers(shared_features_pack, hcls_weight)
        cls_rnn_pad, cls_num_points = rnn_utils.pad_packed_sequence(
            cls_rnn, batch_first=True, padding_value=0
        )
        cls_rnn_index = num_points.cpu().numpy() - 1
        idx = np.arange(0, batch_size_rcnn)
        cls_rnn_end = cls_rnn_pad[idx, cls_rnn_index, :]
        features_res = torch.cat([max_cls_feature, cls_rnn_end, reg_features], dim=-1)

        up_features = features_res[up_mask]
        mid_features = features_res[mid_mask]
        down_features = features_res[down_mask]
        rcnn_cls = torch.zeros(size=[batch_size_rcnn, 1], dtype=torch.float32, device=down_features.device)
        rcnn_reg = torch.zeros(size=[batch_size_rcnn, 7], dtype=torch.float32, device=down_features.device)
        rcnn_cls_up = self.cls_linear_up(up_features)
        rcnn_cls_mid = self.cls_linear_mid(mid_features)
        rcnn_cls_down = self.cls_linear_down(down_features) 
        rcnn_cls[up_mask]=rcnn_cls_up
        rcnn_cls[mid_mask]=rcnn_cls_mid
        rcnn_cls[down_mask]=rcnn_cls_down
        rcnn_reg_up = self.reg_linear_up(up_features)
        rcnn_reg_mid = self.reg_linear_mid(mid_features)
        rcnn_reg_down = self.reg_linear_down(down_features)
        rcnn_reg[up_mask]=rcnn_reg_up
        rcnn_reg[mid_mask]=rcnn_reg_mid
        rcnn_reg[down_mask]=rcnn_reg_down

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict
