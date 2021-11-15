import torch
import torch.nn as nn


class BaseBEVBackboneASCNet(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.base_channel = 128
        self.base_channel2 = 128
        self.base_channel4 = 256
        self.base_channel8 = 512
        self.base_channel16 = 1024
        self.decode_channel = 256

        self.vfe_input = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=self.base_channel, kernel_size=1, stride=1, padding=0, bias=False, groups=1),
            nn.BatchNorm2d(self.base_channel, eps=1e-5, momentum=0.01),
            nn.ReLU(),
        )

        self.feature_scale1 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel, out_channels=self.base_channel2, kernel_size=3, stride=2, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(self.base_channel2, eps=1e-5, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.base_channel2, out_channels=self.base_channel2, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(self.base_channel2, eps=1e-5, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.base_channel2, out_channels=self.base_channel2, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(self.base_channel2, eps=1e-5, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.base_channel2, out_channels=self.base_channel2, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
        )
        self.downscale1 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel, out_channels=self.base_channel2, kernel_size=3, stride=2, padding=1, bias=False, groups=1)
        )
        self.bn_scale1 = nn.Sequential(
            nn.BatchNorm2d(self.base_channel2, eps=1e-5, momentum=0.01),
            nn.ReLU(),
        )
        self.feature_scale2 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel2, out_channels=self.base_channel4, kernel_size=3, stride=2, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(self.base_channel4, eps=1e-5, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.base_channel4, out_channels=self.base_channel4, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(self.base_channel4, eps=1e-5, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.base_channel4, out_channels=self.base_channel4, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(self.base_channel4, eps=1e-5, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.base_channel4, out_channels=self.base_channel4, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
        )
        self.downscale2 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel2, out_channels=self.base_channel4, kernel_size=3, stride=2, padding=1, bias=False, groups=1)
        )
        self.bn_scale2 = nn.Sequential(
            nn.BatchNorm2d(self.base_channel4, eps=1e-5, momentum=0.01),
            nn.ReLU(),
        )

        self.feature_scale3 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel4, out_channels=self.base_channel8, kernel_size=3, stride=2, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(self.base_channel8, eps=1e-5, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.base_channel8, out_channels=self.base_channel8, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(self.base_channel8, eps=1e-5, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.base_channel8, out_channels=self.base_channel8, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(self.base_channel8, eps=1e-5, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.base_channel8, out_channels=self.base_channel8, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
        )

        self.downscale3 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel4, out_channels=self.base_channel8, kernel_size=3, stride=2, padding=1, bias=False, groups=1)
        )
        self.bn_scale3 = nn.Sequential(
            nn.BatchNorm2d(self.base_channel8, eps=1e-5, momentum=0.01),
            nn.ReLU(),
        )
        self.feature_scale4 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel8, out_channels=self.base_channel16, kernel_size=3, stride=2, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(self.base_channel16, eps=1e-5, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.base_channel16, out_channels=self.base_channel16, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(self.base_channel16, eps=1e-5, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.base_channel16, out_channels=self.base_channel16, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(self.base_channel16, eps=1e-5, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.base_channel16, out_channels=self.base_channel16, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
        )
        self.downscale4 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel8, out_channels=self.base_channel16, kernel_size=3, stride=2, padding=1, bias=False, groups=1)
        )
        self.bn_scale4 = nn.Sequential(
            nn.BatchNorm2d(self.base_channel16, eps=1e-5, momentum=0.01),
            nn.ReLU(),
        )
    
        self.feature_scale5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.base_channel16, out_channels=self.base_channel8, kernel_size=2, stride=2, bias=False, groups=1),
            nn.BatchNorm2d(self.base_channel8, eps=1e-5, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.base_channel8, out_channels=self.base_channel8, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(self.base_channel8, eps=1e-5, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.base_channel8, out_channels=self.base_channel8, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(self.base_channel8, eps=1e-5, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.base_channel8, out_channels=self.base_channel8, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
        )
        self.downscale5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.base_channel16, out_channels=self.base_channel8, kernel_size=2, stride=2, bias=False, groups=1)
        )
        self.bn_scale5 = nn.Sequential(
            nn.BatchNorm2d(self.base_channel8, eps=1e-5, momentum=0.01),
            nn.ReLU(),
        )
   
        self.feature_scale6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.base_channel8, out_channels=self.base_channel4, kernel_size=2, stride=2, bias=False, groups=1),
            nn.BatchNorm2d(self.base_channel4, eps=1e-5, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.base_channel4, out_channels=self.base_channel4, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(self.base_channel4, eps=1e-5, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.base_channel4, out_channels=self.base_channel4, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(self.base_channel4, eps=1e-5, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.base_channel4, out_channels=self.base_channel4, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
        )
        self.downscale6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.base_channel8, out_channels=self.base_channel4, kernel_size=2, stride=2, bias=False, groups=1)
        )
        self.bn_scale6 = nn.Sequential(
            nn.BatchNorm2d(self.base_channel4, eps=1e-5, momentum=0.01),
            nn.ReLU(),
        )

        self.feature_scale8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.base_channel4, out_channels=self.base_channel2, kernel_size=2, stride=2, bias=False, groups=1),
            nn.BatchNorm2d(self.base_channel2, eps=1e-5, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.base_channel2, out_channels=self.base_channel2, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(self.base_channel2, eps=1e-5, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.base_channel2, out_channels=self.base_channel2, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(self.base_channel2, eps=1e-5, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.base_channel2, out_channels=self.base_channel2, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
        )
        self.downscale8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.base_channel4, out_channels=self.base_channel2, kernel_size=2, stride=2, bias=False, groups=1)
        )
        self.bn_scale8 = nn.Sequential(
            nn.BatchNorm2d(self.base_channel2, eps=1e-5, momentum=0.01),
            nn.ReLU(),
        )
        self.trans2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.base_channel2, out_channels=self.decode_channel, kernel_size=1, stride=1,  bias=False, groups=1),
            nn.BatchNorm2d(self.decode_channel, eps=1e-5, momentum=0.01),
            nn.ReLU(),
        )
        self.trans4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.base_channel4, out_channels=self.decode_channel, kernel_size=2, stride=2, bias=False, groups=1),
            nn.BatchNorm2d(self.decode_channel, eps=1e-5, momentum=0.01),
            nn.ReLU(),
        )
        self.trans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.base_channel8, out_channels=self.decode_channel, kernel_size=4, stride=4,  bias=False, groups=1),
            nn.BatchNorm2d(self.decode_channel, eps=1e-5, momentum=0.01),
            nn.ReLU(),
        )
        self.num_bev_features = 768
    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features_ori = data_dict['spatial_features']
        spatial_features = self.vfe_input(spatial_features_ori)

        scale1 = self.feature_scale1(spatial_features)
        scale1 = scale1 + self.downscale1(spatial_features)
        scale1 = self.bn_scale1(scale1)
        scale2 = self.feature_scale2(scale1)
        scale2 = scale2 + self.downscale2(scale1)
        scale2 = self.bn_scale2(scale2)
        scale3 = self.feature_scale3(scale2)
        scale3 = scale3 + self.downscale3(scale2)
        scale3 = self.bn_scale3(scale3)
        scale4 = self.feature_scale4(scale3)
        scale4 = scale4 + self.downscale4(scale3)
        scale4 = self.bn_scale4(scale4)

        scale5 = self.feature_scale5(scale4)
        scale5 = scale5 + self.downscale5(scale4)
        scale5 = self.bn_scale5(scale5)

        scale6 = self.feature_scale6(scale5)
        scale6 = scale6 + self.downscale6(scale5)
        scale6 = self.bn_scale6(scale6)

        scale8 = self.feature_scale8(scale6)
        scale8 = scale8 + self.downscale8(scale6)
        scale8 = self.bn_scale8(scale8)

        spatial_features_2X = self.trans2(scale8)
        spatial_features_4X = self.trans4(scale6)
        spatial_features_8X = self.trans8(scale5)

        data_dict['spatial_features_2d'] = torch.cat([spatial_features_2X, spatial_features_4X, spatial_features_8X], dim=1)
        return data_dict
