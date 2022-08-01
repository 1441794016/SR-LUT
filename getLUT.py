from tkinter import Scale
import torch
import numpy as np
from train_SR import test, compute_sr
from model.SR import SRNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_pixel = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255] # 一共17个采样像素点 [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]
scale = 4 # 放大倍数
LUT = np.zeros([len(sample_pixel), len(sample_pixel), len(sample_pixel), len(sample_pixel), scale, scale])
inputs = torch.zeros(1, 1, 2, 2)

test_model = SRNet(scale)
test_model = test_model.to(device)
test_model.eval()
with torch.no_grad():
    for i in sample_pixel:
        # print(sample_pixel.index(i))
        for j in sample_pixel:
            for f in sample_pixel:
                for k in sample_pixel:
                    inputs = torch.FloatTensor([[[[i, j],
                                            [f, k]]]])
                    # inputs:(1, 1, 2, 2), 代表2x2的感受野
                    inputs = inputs.to(device)
                    LUT[sample_pixel.index(i), sample_pixel.index(j), sample_pixel.index(f), sample_pixel.index(k)] = test_model(inputs).squeeze().cpu().numpy()

np.save('LUT.npy',LUT)

print(LUT[0, 0, 0, 0])
print("over!")

