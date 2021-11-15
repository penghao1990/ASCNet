import numpy as np
import torch.nn as nn
import torch
from torchvision import transforms
from .anchor_head_template import AnchorHeadTemplate
import matplotlib.pyplot as plt


class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )
        # self.conv_center = nn.Conv2d(
        #     input_channels, 1, kernel_size=1
        # )
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)
        # nn.init.constant_(self.conv_center.bias, -np.log((1 - pi) / pi))

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        # reg_features_2d = data_dict['reg_features_2d']
        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)
        # center_preds = self.conv_center(spatial_features_2d)
        # cls_preds = cls_preds+center_preds[:, 0, :, :].unsqueeze(1)
        # center_mask = center_preds[:, 0, :, :] < center_preds[:, 1, :, :]
        # pos = center_preds[:, 0, :, :]
        # pos[center_mask] = -10
        # pos_line = pos.view(-1)
        # val, index = torch.topk(pos_line, 5)
        # mask = pos < val.min()
        # pos[mask] = -10
        # center_max = torch.max(center_preds, 1)[1]
        # plot_data = center_preds[0, 0, :, :]
        # plot_data = center_preds[0, 0, :, :]+cls_preds[0, 0, :, :]
        # plt.imshow(plot_data.cpu().numpy())
        # plt.colorbar()
        # plt.show()
        # center_max = torch.max(center_preds, 1)[0]
        # unloader = transforms.ToPILImage()
        # image = center_preds[1, 0, :, :].cpu().clone()  # clone the tensor
        # image = image.squeeze(0)  # remove the fake batch dimension
        # image = unloader(image)
        # image.save('example.jpg')

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        # center_preds = center_preds.permute(0, 2, 3, 1).contiguous()

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        # self.forward_ret_dict['center_preds'] = center_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False
            # self.forward_ret_dict.update({
                # 'proposal_cls_preds': batch_cls_preds,
                # 'proposal_box_preds': batch_box_preds,
                # 'pillar_features_loss': data_dict['pillar_features_loss']
            # })

        return data_dict
