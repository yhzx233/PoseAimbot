import logging

import torch
from torch.utils.data import DataLoader

from src.utils.data_transformer import COCOKeyPoints
from src.model import KeypointModel

def train_model(model: KeypointModel, device: torch.device, epochs: int = 10, batch_size: int = 10, save_checkpoint: bool = True):
    # load dataset
    train_dataset = COCOKeyPoints(root="data", annFile="data/annotations/person_keypoints_train2017.json")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # loss function
    def criterion(output, target, mask):
        loss = torch.sum((output - target) ** 2 * mask) / torch.sum(mask)
        return loss

    # record loss in loss.txt
    loss_file = open("loss.txt", "w")

    # optimizer
    base_lr = 0.01
    optimizer = torch.optim.SGD(
        [
            {"params": model.vgg.parameters(), "lr": base_lr},
            {"params": model.cpm_1.parameters(), "lr": base_lr*2},
            {"params": model.cpm_2.parameters(), "lr": base_lr*2},
            {"params": model.cpm_3.parameters(), "lr": base_lr*2},
        ],
        lr=base_lr*2, momentum=0.9, weight_decay=5e-4)

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.33)

    model.train()

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            out1, out2, out3 = model(data)
            # add mask
            mask = target[:, -1:, :, :]
            target = target[:, :-1, :, :]

            loss1 = criterion(out1, target, mask)
            loss2 = criterion(out2, target, mask)
            loss3 = criterion(out3, target, mask)

            loss = loss1 + loss2 + loss3

            loss.backward()
            optimizer.step()
            scheduler.step()

            # 3 loss
            loss_file.write(f"{loss1.item()} {loss2.item()} {loss3.item()} {loss.item()}\n")
            loss_file.flush()

            if batch_idx % 10 == 0:
                # print 3 loss
                logging.info(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\t \
                Loss1: {loss1.item():.6f}\tLoss2: {loss2.item():.6f}\tLoss3: {loss3.item():.6f}\tLoss: {loss.item():.6f}")

            if batch_idx % 5000 == 0:
                logging.info(f"Current learning rate: {scheduler.get_last_lr()}")
                # torch.save(model.state_dict(), f"model/pre_model_{batch_idx}.pth")
        
        if save_checkpoint:
            torch.save(model.state_dict(), f"model/model_{epoch}.pth")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    # model = KeypointModel()
    # model.load_state_dict(torch.load("model/mask_model_2000.pth"))

    model = KeypointModel(pretrained=True)

    model.to(device)

    train_model(model, device)