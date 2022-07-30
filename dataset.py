import h5py
import numpy as np
import random
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def random_crop(hr, lr, size, scale):
    # 随机裁剪, size为裁剪的大小 scale为放大的倍数
    # lr 为 ：h w c格式
    lr_h, lr_w = lr.shape[:-1]
    x = random.randint(0, lr_w - size)
    y = random.randint(0, lr_h - size)

    crop_lr = lr[y:y + size, x: x + size].copy()
    crop_hr = hr[y * scale:(y + size) * scale, x * scale:(x + size) * scale]

    return crop_hr, crop_lr


class TrainDataset(data.Dataset):
    def __init__(self, path, size, scale):
        # size为裁剪的尺寸
        super(TrainDataset, self).__init__()

        h5f = h5py.File(path, 'r')

        self.size = size
        self.scale = scale
        self.hr = [v[:] for v in h5f["HR"].values()]
        self.lr = [[v[:] for v in h5f["X{}".format(scale)].values()]]

        h5f.close()

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, item):
        img_data = [(self.hr[item], self.lr[i][item]) for i, _ in enumerate(self.lr)]
        img_data = [random_crop(hr, lr, self.size, self.scale) for i, (hr, lr) in enumerate(img_data)]
        return [(self.transform(hr), self.transform(lr)) for hr, lr in img_data]

    def __len__(self):
        return len(self.hr)


class TestDataset(data.Dataset):
    def __init__(self, path, size, scale):
        super(TestDataset, self).__init__()

        h5f = h5py.File(path, 'r')

        self.size = size
        self.scale = scale

        self.hr = [v[:] for v in h5f["original"].values()]
        self.lr = [v[:] for v in h5f["LRbicx{}".format(scale)].values()]

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, item):
        hr = self.hr[item]
        lr = self.lr[item]

        return self.transform(hr), self.transform(lr)

    def __len__(self):
        return len(self.hr)


if __name__ == "__main__":
    path = "DIV2K_train/DIV2K_train.h5"
    size = 64
    scale = 4
    batch_size = 2

    dataset = TrainDataset(path, size, scale)
    print(len(dataset.hr))
    # 800
    print(dataset.hr[0].shape)
    # (1404, 2040, 3)
    print(type(dataset.lr[0]))
    # <class 'list'>
    print(len(dataset.lr))
    # 1
    print(len(dataset.lr[0]))
    # 800
    print(dataset.lr[0][0].shape)
    # (351, 510, 3) 格式为 h w c
    # data_loader = data.DataLoader(dataset, batch_size=2, num_workers=0)