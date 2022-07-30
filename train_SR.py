import torch
import numpy as np
import time
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.SR import SRNet
from dataset import TrainDataset, TestDataset
from matplotlib import pyplot as plt
from PIL import Image
import math


def compute_sr(model, lr_channel):
    sr_channel_rot0 = model(F.pad(lr_channel, (0, 1, 0, 1), mode='reflect'))

    sr_channel_rot90 = model(F.pad(torch.rot90(lr_channel, 1, [2, 3]), (0, 1, 0, 1), mode='reflect'))
    sr_channel_rot90 = torch.rot90(sr_channel_rot90, 3, [2, 3])

    sr_channel_rot180 = model(F.pad(torch.rot90(lr_channel, 2, [2, 3]), (0, 1, 0, 1), mode='reflect'))
    sr_channel_rot180 = torch.rot90(sr_channel_rot180, 2, [2, 3])

    sr_channel_rot270 = model(F.pad(torch.rot90(lr_channel, 3, [2, 3]), (0, 1, 0, 1), mode='reflect'))
    sr_channel_rot270 = torch.rot90(sr_channel_rot270, 1, [2, 3])

    return (sr_channel_rot0 + sr_channel_rot90 + sr_channel_rot180 + sr_channel_rot270) / 4


def PSNR(img1, img2):
    temp = img1 - img2
    temp = temp.view(-1)
    mse = torch.mean(temp ** 2)
    if mse == 0:
        return 100
    pixel_max = 255.0
    return 20 * math.log10(pixel_max / math.sqrt(mse))


def test(model, dataloader, device):
    model.eval()
    test_psnr = 0
    with torch.no_grad():
        for item, inputs in enumerate(dataloader):
            hr, lr = inputs[0], inputs[1]
            hr = hr.to(device)
            lr = lr.to(device)

            lr_channel1 = lr[:, 0:1, :, :]
            lr_channel2 = lr[:, 1:2, :, :]
            lr_channel3 = lr[:, 2:3, :, :]

            sr_channel1 = compute_sr(model, lr_channel1)
            sr_channel2 = compute_sr(model, lr_channel2)
            sr_channel3 = compute_sr(model, lr_channel3)

            sr = torch.cat((sr_channel1, sr_channel2, sr_channel3), dim=1)
            b, c, h, w = sr.shape
            hr = hr.resize_(b, c, h, w)  # 将hr调整至和sr一样的大小， 才可以计算PSNR
            # print("srshape", str(sr.shape))
            # print("hrshape", str(hr.shape))
            for number in range(sr.shape[0]):
                # print(sr[number].shape)
                # print(hr[number].shape)
                test_psnr += PSNR(sr[number], hr[number])
                img = sr[number].mul(255).clamp(0, 255)
                # print(sr[number])
                img = img.detach().cpu().numpy().transpose(1, 2, 0)  # chw -> hwc
                img = Image.fromarray(np.uint8(img)).convert('RGB')
                img.save("test/{}_{}.png".format(item, number))
    return test_psnr


if __name__ == "__main__":
    batch_size = 32
    iteration = 2e5
    scale = 4  # 放大的倍数
    size = 48  # 裁剪的大小
    train_path = "DIV2K_train/DIV2K_train.h5"
    test_path = "Set14_test/Set14_test.h5"

    sr_model = SRNet(r=scale)
    train_dataset = TrainDataset(train_path, size, scale)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=0,
                                  shuffle=True)
    test_dataset = TestDataset(test_path, size, scale)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 num_workers=0,
                                 shuffle=False)

    lr = 0.0004
    loss_func = torch.nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(sr_model.parameters(), lr=lr)
    sr_model = sr_model.to(device)

    print_interval = 1000
    step = 0
    all_psnr = []
    all_loss = []

    status = True
    while status:
        for item, inputs in enumerate(train_dataloader):
            sr_model.train()
            start_time = time.time()
            hr, lr = inputs[-1][0], inputs[-1][1]
            # print(hr.shape)
            # torch.Size([32, 3, 192, 192])
            # print(lr.shape)
            # torch.Size([32, 3, 48, 48])
            hr = hr.to(device)
            lr = lr.to(device)

            # 三个通道单独处理
            lr_channel1 = lr[:, 0:1, :, :]
            lr_channel2 = lr[:, 1:2, :, :]
            lr_channel3 = lr[:, 2:3, :, :]
            # print(lr_channel1.shape)
            # torch.Size([32, 1, 48, 48])
            sr_channel1 = compute_sr(sr_model, lr_channel1)
            # print(sr_channel1.shape)
            # torch.Size([32, 1, 192, 192])
            sr_channel2 = compute_sr(sr_model, lr_channel2)
            sr_channel3 = compute_sr(sr_model, lr_channel3)

            # 将每个通道重新合并起来
            sr = torch.cat((sr_channel1, sr_channel2, sr_channel3), dim=1)

            loss = loss_func(sr, hr)
            all_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step = step + 1
            save_dict_path = 'best.pth'
            if step % print_interval == 0:
                psnr = test(sr_model, test_dataloader, device)
                if all_psnr:
                    all_psnr.append(psnr)
                    if psnr > max(all_psnr[0:len(all_psnr) - 1]):
                        sr_model.eval()
                        torch.save(sr_model.state_dict(), save_dict_path)
                else:
                    all_psnr.append(psnr)
                    torch.save(sr_model.state_dict(), save_dict_path)
                print("psnr: ", str(psnr))

            end_time = time.time()

            print(str(step), "/", str(iteration), "loss:", str(loss.item()))
            print("time costing: ", str(end_time - start_time))

            if step >= iteration:
                status = False
                break

    plt.subplot(1, 2, 1)
    plt.plot(all_loss)
    plt.title("train loss")
    plt.xlabel('iteration')
    plt.ylabel("MSE loss")
    plt.savefig('SRtrainloss.png')
    plt.subplot(1, 2, 2)
    plt.plot(all_psnr)
    plt.xlabel('iteration x 100')
    plt.title("test psnr")
    plt.ylabel("PSNR")
    plt.savefig('SRtest.png')

    print(all_psnr)
