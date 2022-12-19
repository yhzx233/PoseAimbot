from typing import List, Tuple

import torch
from torch import Tensor

class Heatmapper:

    def __init__(self, height: int, width: int, sigma: float = 1.0) -> None:
        self.height = height
        self.width = width
        self.sigma = sigma

        x_range = torch.arange(0, height).float()
        y_range = torch.arange(0, width).float()
        x_grid, y_grid = torch.meshgrid(x_range, y_range, indexing="ij")
        x_grid, y_grid = x_grid + 0.5, y_grid + 0.5

        self.x_grid = x_grid
        self.y_grid = y_grid

    def __call__(self, map: Tensor, keypoints: List[Tuple[int, int]]) -> Tensor:
        # make sure map is the right size
        assert map.shape == (self.height, self.width)

        for x, y in keypoints:
            map = torch.maximum(map, self._gaussian(x, y))
        return map
    
    def _gaussian(self, x: int, y: int) -> Tensor:

        sigma_2sq = 2 * self.sigma ** 2

        return torch.exp(-((self.x_grid - x) ** 2 + (self.y_grid - y) ** 2) / sigma_2sq)

        

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    target = torch.zeros( (46, 46) )
    print(target)

    heatmapper = Heatmapper(46, 46, sigma=2.0)

    target = heatmapper(target, [(5.1, 1.4), (11.3, 12.7)])
    
    plt.imshow(target, cmap="coolwarm")
    plt.show()
