from typing import List, Tuple

import numpy as np

import torch
from torchvision import transforms
from torchvision.transforms import functional as F

from src.model import KeypointModel

class Keypoint:

    def __init__(self, model_path: str, half: bool = False, threshold: float = 0.1) -> None:
        self.model = KeypointModel(pretrained=False)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.half = half
        if half:
            self.model.half()

        self.threshold = threshold

        self.upSample = torch.nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.gaussian_filter = transforms.GaussianBlur(3, sigma=3.0)

        if torch.cuda.is_available():
            self.model.cuda()
            self.upSample.cuda()
            self.gaussian_filter.cuda()

    def __call__(self, img: np.array) -> List[Tuple[float, float]]:

        # BGR numpy to RGB normalized tensor
        x = torch.from_numpy(img).permute(2, 0, 1)

        if torch.cuda.is_available():
            x = x.cuda()

        if self.half:
            x = x.half()
        x = x[[2, 1, 0], :, :] / 255.0 - 0.5
        x = x.unsqueeze(0)

        # 缩放到最长边为368
        scale = 368 / max(img.shape[:2])
        x = F.resize(x, (int(img.shape[0] * scale), int(img.shape[1] * scale)))

        with torch.no_grad():
            x = self.model(x)
            x = self.upSample(x)
            x = self.gaussian_filter(x)

        import matplotlib.pyplot as plt

        # 显示heatmap
        # for i in range(0, 17):
        #     plt.imshow(x[0, i].cpu().numpy(), cmap="coolwarm")
        #     plt.show()
        

        # get all the peaks
        all_peaks = []

        map_ori = x[0]
        map_left = torch.zeros_like(map_ori)
        map_right = torch.zeros_like(map_ori)
        map_up = torch.zeros_like(map_ori)
        map_down = torch.zeros_like(map_ori)

        map_left[:, 1:, :] = map_ori[:, :-1, :]
        map_right[:, :-1, :] = map_ori[:, 1:, :]
        map_up[:, :, 1:] = map_ori[:, :, :-1]
        map_down[:, :, :-1] = map_ori[:, :, 1:]

        peaks_binary = (map_ori >= map_left) & (map_ori >= map_right) & (map_ori >= map_up) & (map_ori >= map_down) & (map_ori > self.threshold)

        for i in range(x.shape[1]-1):
            non_zero_p = torch.nonzero(peaks_binary[i]).cpu().numpy()
            peaks = zip(non_zero_p[:, 1], non_zero_p[:, 0])
            peaks_with_score = [(x[0] / scale, x[1] / scale, map_ori[i, x[1], x[0]], ) for x in peaks]
            peaks_with_score = sorted(peaks_with_score, key=lambda x: x[2], reverse=True)
            all_peaks.append(peaks_with_score)
        
        return all_peaks

        