from typing import Any, Callable, Optional, Tuple, List

import torch
from torch import Tensor
from torchvision.datasets.coco import CocoDetection
from torchvision.transforms import ToTensor, RandomCrop
import torchvision.transforms.functional as F

from PIL import Image

from src.utils.heatmapper import Heatmapper

class DataTransformer:

    def __init__(self) -> None:
        self.heatmapper = Heatmapper(46, 46)

    def __call__(self, image: Image.Image, anns: List[Any]) -> Tuple[Tensor, Tensor]:    
        img_size = 368
        num_parts = 17
        stride = 8

        # transform image
        image = ToTensor()(image) - 0.5
        # image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # if image is smaller than img_size, pad it
        if image.shape[1] < img_size or image.shape[2] < img_size:
            pad_h = max(0, img_size - image.shape[1])
            pad_w = max(0, img_size - image.shape[2])
            image = F.pad(image, (0, 0, pad_w, pad_h), fill=0.0, padding_mode="constant")

        # random crop, i, j, h, w
        crop_params = RandomCrop.get_params(image, output_size=(img_size, img_size))
        image = F.crop(image, *crop_params)
        
        # 17 keypoints + 1 background + 1 mask
        target = torch.zeros((num_parts+2, img_size//stride, img_size//stride))
        target[num_parts+1] = 0

        # all coordinates of each kind of keypoints
        keypoints = [[] for _ in range(num_parts)]
        
        for ann in anns:
            if ann["iscrowd"]:
                continue
            x, y, w, h = ann["bbox"]
            x, y, w, h = (x-crop_params[1])//stride, (y-crop_params[0])//stride, w//stride, h//stride
            x, y, w, h = max(0, x), max(0, y), min(img_size//stride, x+w), min(img_size//stride, y+h)
            target[num_parts+1, int(y):int(h), int(x):int(w)] = 1

            ann_keypoints = ann["keypoints"]
            x, y, v = ann_keypoints[0::3], ann_keypoints[1::3], ann_keypoints[2::3]

            for i in range(num_parts):
                if v[i] != 0:
                    keypoints[i].append((y[i], x[i]))

        for ann in anns:
            if ann["iscrowd"] == 0:
                continue
            x, y, w, h = ann["bbox"]
            x, y, w, h = (x-crop_params[1])//stride, (y-crop_params[0])//stride, w//stride, h//stride
            x, y, w, h = max(0, x), max(0, y), min(img_size//stride, x+w), min(img_size//stride, y+h)
            target[num_parts+1, int(y):int(h), int(x):int(w)] = 0

        # crop and scale keypoints
        for i in range(num_parts):
            for j in range(len(keypoints[i])):
                keypoints[i][j] = (keypoints[i][j][0]-crop_params[0])/stride, (keypoints[i][j][1]-crop_params[1])/stride
        
        for i in range(num_parts):
            target[i] = self.heatmapper(target[i], keypoints[i])

        # calculate background
        target[num_parts] = 1 - torch.max(target[:num_parts], dim=0)[0]

        # enhance heatmap
        # target[:num_parts+1] *= 4

        return image, target

class COCOKeyPoints(CocoDetection):

    def __init__(self, root: str, annFile: str) -> None:
        super().__init__(root, annFile, transforms=DataTransformer())

        # remove images without keypoints
        self.ids = [id for id in self.ids if self.coco.loadAnns(self.coco.getAnnIds(imgIds=id, iscrowd=None))]

if __name__ == "__main__":
    coco_dataset = COCOKeyPoints(root="data", annFile="data/annotations/person_keypoints_train2017.json")

    import matplotlib.pyplot as plt

    # 遍历数据集，展示有mask的图片
    for i in range(200, 300):
        img, target = coco_dataset[i]

        # 如果mask全为1，说明没有mask
        if torch.sum(target[-1]) == 46*46:
            continue

        # 拼接img和target
        img = img.permute(1, 2, 0).numpy()
        print(img)
        target = target.numpy()

        # 画图
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img)
        ax[1].imshow(target[-1], cmap="coolwarm")

        plt.show()