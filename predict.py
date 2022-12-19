import torch

from src.model import KeypointModel
from src.utils.data_transformer import COCOKeyPoints

model = KeypointModel(pretrained=False)
model.load_state_dict(torch.load("model/model_2.pth"))

model.train()

train_dataset = COCOKeyPoints(root="data", annFile="data/annotations/person_keypoints_train2017.json")

for i in range(0, 100):
    img, target = train_dataset[i]

    out = model(img.unsqueeze(0))

    import matplotlib.pyplot as plt

    # 拼接img和target和out
    img = (img + 0.5).permute(1, 2, 0).numpy()
    target = target.numpy()
    out = torch.stack(out).detach().numpy().squeeze()

    # 画图
    fig, ax = plt.subplots(1, 5, figsize=(20, 5))
    ax[0].imshow(img)
    ax[1].imshow(target[-2], cmap="coolwarm")
    ax[2].imshow(out[0][-1], cmap="coolwarm")
    ax[3].imshow(out[1][-1], cmap="coolwarm")
    ax[4].imshow(out[2][-1], cmap="coolwarm")

    plt.show()