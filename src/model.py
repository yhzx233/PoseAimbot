import torch
import torch.nn as nn

from torch import Tensor

from torchvision.models.vgg import make_layers as make_layers_vgg
from torchvision.models.vgg import vgg19_bn

from typing import List

def make_layers_cpm(in_channels: int, cfg: List[int], kernel_size: int, conv1x1: int, out_channels: int) -> nn.Sequential:
    layers = []
    for v in cfg:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=kernel_size, padding=kernel_size//2)
        layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    layers += [nn.Conv2d(in_channels, conv1x1, kernel_size=1), nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(conv1x1, out_channels, kernel_size=1)]

    return nn.Sequential(*layers)


class KeypointModel(nn.Module):

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()

        # VGG-19 first 10 layers
        self.vgg = make_layers_vgg([64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 256, 128], batch_norm=True)

        num_parts = 18
        
        # Convolution Pose Machine
        self.cpm_1 = make_layers_cpm(128, [128, 128, 128], 3, 512, num_parts)
        self.cpm_2 = make_layers_cpm(128 + num_parts, [128] * 3, 3, 128, num_parts)
        self.cpm_3 = make_layers_cpm(128 + num_parts, [128] * 3, 3, 128, num_parts)

        if pretrained:
            vgg19 = vgg19_bn()
            vgg19.load_state_dict(torch.load("model/vgg19_bn-c79401a0.pth"))

            for i in range(33):
                self.vgg[i].load_state_dict(vgg19.features[i].state_dict())

            # body_pose = torch.load("model/body_pose_model.pth")
            # keys = ["conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2", "conv3_3", "conv3_4", "conv4_1", "conv4_2", "conv4_3_CPM", "conv4_4_CPM"]
            # vgg_id = 0

            # for key in keys:
            #     while self.vgg[vgg_id].__class__.__name__ != "Conv2d":
            #         vgg_id += 1
            #     self.vgg[vgg_id].weight.data = body_pose[key + ".weight"]
            #     self.vgg[vgg_id].bias.data = body_pose[key + ".bias"]
            #     vgg_id += 1

            # cpm_id = 0
            # for key in ["conv5_1_CPM_L2", "conv5_2_CPM_L2", "conv5_3_CPM_L2", "conv5_4_CPM_L2", "conv5_5_CPM_L2"]:
            #     while self.cpm_1[cpm_id].__class__.__name__ != "Conv2d":
            #         cpm_id += 1
            #     self.cpm_1[cpm_id].weight.data = body_pose[key + ".weight"]
            #     self.cpm_1[cpm_id].bias.data = body_pose[key + ".bias"]
            #     cpm_id += 1


    def forward(self, x: Tensor) -> Tensor:
        out0 = self.vgg(x)
        out1 = self.cpm_1(out0)
        out2 = self.cpm_2(torch.cat([out0, out1], dim=1))
        out3 = self.cpm_3(torch.cat([out0, out2], dim=1))


        if self.training:
            return out1, out2, out3
        else:
            return out3

if __name__ == "__main__":
    model = KeypointModel(pretrained=True)
    print(model)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y[0].shape, y[1].shape, y[2].shape)