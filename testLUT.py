import os
import numpy as np
import math
from PIL import Image



def tetrahedral_interpolation(LUT, I0, I1, I2, I3, W):
    MSB0 = int(I0 / 16)
    MSB1 = int(I1 / 16)
    MSB2 = int(I2 / 16)
    MSB3 = int(I3 / 16)

    LSB0 = I0 - MSB0 * 16
    LSB1 = I1 - MSB1 * 16
    LSB2 = I2 - MSB2 * 16
    LSB3 = I3 - MSB3 * 16
    
    P0000 = LUT[MSB0][MSB1][MSB2][MSB3]
    if LSB0 > LSB1 > LSB2 > LSB3:
        w0 = W - LSB0
        w1 = LSB0 - LSB1
        w2 = LSB1 - LSB2
        w3 = LSB2 - LSB3
        w4 = LSB3
        O1 = LUT[MSB0 + 1][MSB1][MSB2][MSB3]
        O2 = LUT[MSB0 + 1][MSB1 + 1][MSB2][MSB3]
        O3 = LUT[MSB0 + 1][MSB1 + 1][MSB2 + 1][MSB3]
    if LSB0 > LSB1 > LSB3 > LSB2:
        w0 = W - LSB0
        w1 = LSB0 - LSB1
        w2 = LSB1 - LSB3
        w3 = LSB3 - LSB2
        w4 = LSB2
        O1 = LUT[MSB0 + 1][MSB1][MSB2][MSB3]
        O2 = LUT[MSB0 + 1][MSB1 + 1][MSB2][MSB3]
        O3 = LUT[MSB0 + 1][MSB1 + 1][MSB2][MSB3 + 1]
    if LSB0 > LSB3 > LSB1 > LSB2:
        w0 = W - LSB0
        w1 = LSB0 - LSB3
        w2 = LSB3 - LSB1
        w3 = LSB1 - LSB2
        w4 = LSB2
        O1 = LUT[MSB0 + 1][MSB1][MSB2][MSB3]
        O2 = LUT[MSB0 + 1][MSB1][MSB2][MSB3 + 1]
        O3 = LUT[MSB0 + 1][MSB1 + 1][MSB2][MSB3 + 1]
    if LSB3 > LSB0 > LSB1 > LSB2:
        w0 = W - LSB3
        w1 = LSB3 - LSB0
        w2 = LSB0 - LSB1
        w3 = LSB1 - LSB2
        w4 = LSB2
        O1 = LUT[MSB0][MSB1][MSB2][MSB3 + 1]
        O2 = LUT[MSB0 + 1][MSB1][MSB2][MSB3 + 1]
        O3 = LUT[MSB0 + 1][MSB1 + 1][MSB2][MSB3 + 1]
    if LSB0 > LSB2 > LSB1 > LSB3:
        w0 = W - LSB0
        w1 = LSB0 - LSB2
        w2 = LSB2 - LSB1
        w3 = LSB1 - LSB3
        w4 = LSB3
        O1 = LUT[MSB0 + 1][MSB1][MSB2][MSB3]
        O2 = LUT[MSB0 + 1][MSB1][MSB2 + 1][MSB3]
        O3 = LUT[MSB0 + 1][MSB1 + 1][MSB2 + 1][MSB3]
    if LSB0 > LSB2 > LSB3 > LSB1:
        w0 = W - LSB0
        w1 = LSB0 - LSB2
        w2 = LSB2 - LSB3
        w3 = LSB3 - LSB1
        w4 = LSB1
        O1 = LUT[MSB0 + 1][MSB1][MSB2][MSB3]
        O2 = LUT[MSB0 + 1][MSB1][MSB2 + 1][MSB3]
        O3 = LUT[MSB0 + 1][MSB1][MSB2 + 1][MSB3 + 1]
    if LSB0 > LSB3 > LSB2 >LSB1:
        w0 = W - LSB0
        w1 = LSB0 - LSB3
        w2 = LSB3 - LSB2
        w3 = LSB2 - LSB1
        w4 = LSB1
        O1 = LUT[MSB0 + 1][MSB1][MSB2][MSB3]
        O2 = LUT[MSB0 + 1][MSB1][MSB2][MSB3 + 1]
        O3 = LUT[MSB0 + 1][MSB1][MSB2 + 1][MSB3 + 1]
    if LSB3 > LSB0 > LSB2 > LSB1:
        w0 = W - LSB3
        w1 = LSB3 - LSB0
        w2 = LSB0 - LSB2
        w3 = LSB2 - LSB1
        w4 = LSB1
        O1 = LUT[MSB0][MSB1][MSB2][MSB3 + 1]
        O2 = LUT[MSB0 + 1][MSB1][MSB2][MSB3 + 1]
        O3 = LUT[MSB0 + 1][MSB1][MSB2 + 1][MSB3 + 1]
    if LSB2 > LSB0 > LSB1 >LSB3:
        w0 = W - LSB2
        w1 = LSB2 - LSB0
        w2 = LSB0 - LSB1
        w3 = LSB1 - LSB3
        w4 = LSB3
        O1 = LUT[MSB0][MSB1][MSB2 + 1][MSB3]
        O2 = LUT[MSB0 + 1][MSB1][MSB2 + 1][MSB3]
        O3 = LUT[MSB0 + 1][MSB1 + 1][MSB2 + 1][MSB3]
    if LSB2 > LSB0 > LSB3 > LSB1:
        w0 = W - LSB2
        w1 = LSB2 - LSB0
        w2 = LSB0 - LSB3
        w3 = LSB3 - LSB1
        w4 = LSB1
        O1 = LUT[MSB0][MSB1][MSB2 + 1][MSB3]
        O2 = LUT[MSB0 + 1][MSB1][MSB2 + 1][MSB3]
        O3 = LUT[MSB0 + 1][MSB1][MSB2 + 1][MSB3 + 1]
    if LSB2 > LSB3 > LSB0 > LSB1:
        w0 = W - LSB2
        w1 = LSB2 - LSB3
        w2 = LSB3 - LSB0
        w3 = LSB0 - LSB1
        w4 = LSB1
        O1 = LUT[MSB0][MSB1][MSB2 + 1][MSB3]
        O2 = LUT[MSB0][MSB1][MSB2 + 1][MSB3 + 1]
        O3 = LUT[MSB0 + 1][MSB1][MSB2 + 1][MSB3 + 1]
    if LSB3 > LSB2 > LSB0 > LSB1:
        w0 = W - LSB3
        w1 = LSB3 - LSB2
        w2 = LSB2 - LSB0
        w3 = LSB0 - LSB1
        w4 = LSB1
        O1 = LUT[MSB0][MSB1][MSB2][MSB3 + 1]
        O2 = LUT[MSB0][MSB1][MSB2 + 1][MSB3 + 1]
        O3 = LUT[MSB0 + 1][MSB1][MSB2 + 1][MSB3 + 1]
    if LSB1 > LSB0 > LSB2 > LSB3:
        w0 = W - LSB1
        w1 = LSB1 - LSB0
        w2 = LSB0 - LSB2
        w3 = LSB2 - LSB3
        w4 = LSB3
        O1 = LUT[MSB0][MSB1 + 1][MSB2][MSB3]
        O2 = LUT[MSB0 + 1][MSB1 + 1][MSB2][MSB3]
        O3 = LUT[MSB0 + 1][MSB1 + 1][MSB2 + 1][MSB3]
    if LSB1 > LSB0 > LSB3 >LSB2:
        w0 = W - LSB1
        w1 = LSB1 - LSB0
        w2 = LSB0 - LSB3
        w3 = LSB3 - LSB2
        w4 = LSB2
        O1 = LUT[MSB0][MSB1 + 1][MSB2][MSB3]
        O2 = LUT[MSB0 + 1][MSB1 + 1][MSB2][MSB3]
        O3 = LUT[MSB0 + 1][MSB1 + 1][MSB2][MSB3 + 1]
    if LSB1 > LSB3 > LSB0 > LSB2:
        w0 = W - LSB1
        w1 = LSB1 - LSB3
        w2 = LSB3 - LSB0
        w3 = LSB0 - LSB2
        w4 = LSB2
        O1 = LUT[MSB0][MSB1 + 1][MSB2][MSB3]
        O2 = LUT[MSB0][MSB1 + 1][MSB2][MSB3 + 1]
        O3 = LUT[MSB0 + 1][MSB1 + 1][MSB2][MSB3 + 1]
    if LSB3 > LSB1 > LSB0 > LSB2:
        w0 = W - LSB3
        w1 = LSB3 - LSB1
        w2 = LSB1 - LSB0
        w3 = LSB0 - LSB2
        w4 = LSB2
        O1 = LUT[MSB0][MSB1][MSB2][MSB3 + 1]
        O2 = LUT[MSB0][MSB1 + 1][MSB2][MSB3 + 1]
        O3 = LUT[MSB0 + 1][MSB1 + 1][MSB2][MSB3 + 1]
    if LSB1 > LSB2 > LSB0 > LSB3:
        w0 = W - LSB1
        w1 = LSB1 - LSB2
        w2 = LSB2 - LSB0
        w3 = LSB0 - LSB3
        w4 = LSB3
        O1 = LUT[MSB0][MSB1 + 1][MSB2][MSB3]
        O2 = LUT[MSB0][MSB1 + 1][MSB2 + 1][MSB3]
        O3 = LUT[MSB0 + 1][MSB1 + 1][MSB2 + 1][MSB3]
    if LSB1 > LSB2 > LSB3 > LSB0:
        w0 = W - LSB1
        w1 = LSB1 - LSB2
        w2 = LSB2 - LSB3
        w3 = LSB3 - LSB0
        w4 = LSB0
        O1 = LUT[MSB0][MSB1 + 1][MSB2][MSB3]
        O2 = LUT[MSB0][MSB1 + 1][MSB2 + 1][MSB3]
        O3 = LUT[MSB0][MSB1 + 1][MSB2 + 1][MSB3 + 1]
    if LSB1 > LSB3 > LSB2 > LSB0:
        w0 = W - LSB1
        w1 = LSB1 - LSB3
        w2 = LSB3 - LSB2
        w3 = LSB2 - LSB0
        w4 = LSB0
        O1 = LUT[MSB0][MSB1 + 1][MSB2][MSB3]
        O2 = LUT[MSB0][MSB1 + 1][MSB2][MSB3 + 1]
        O3 = LUT[MSB0][MSB1 + 1][MSB2 + 1][MSB3 + 1]
    if LSB3 > LSB1 > LSB2 > LSB0:
        w0 = W - LSB3
        w1 = LSB3 - LSB1
        w2 = LSB1 - LSB2
        w3 = LSB2 - LSB0
        w4 = LSB0
        O1 = LUT[MSB0][MSB1][MSB2][MSB3 + 1]
        O2 = LUT[MSB0][MSB1 + 1][MSB2][MSB3 + 1]
        O3 = LUT[MSB0 + 1][MSB1 + 1][MSB2][MSB3 + 1]
    if LSB2 > LSB1 > LSB0 > LSB3:
        w0 = W - LSB2
        w1 = LSB2 - LSB1
        w2 = LSB1 - LSB0
        w3 = LSB0 - LSB3
        w4 = LSB3
        O1 = LUT[MSB0][MSB1][MSB2 + 1][MSB3]
        O2 = LUT[MSB0][MSB1 + 1][MSB2 + 1][MSB3]
        O3 = LUT[MSB0 + 1][MSB1 + 1][MSB2 + 1][MSB3]
    if LSB2 > LSB1 > LSB3 > LSB0:
        w0 = W - LSB2
        w1 = LSB2 - LSB1
        w2 = LSB1 - LSB3
        w3 = LSB3 - LSB0
        w4 = LSB0
        O1 = LUT[MSB0][MSB1][MSB2 + 1][MSB3]
        O2 = LUT[MSB0][MSB1 + 1][MSB2 + 1][MSB3]
        O3 = LUT[MSB0][MSB1 + 1][MSB2 + 1][MSB3 + 1]
    if LSB2 > LSB3 > LSB1 > LSB0:
        w0 = W - LSB2
        w1 = LSB2 - LSB3
        w2 = LSB3 - LSB1
        w3 = LSB1 - LSB0
        w4 = LSB0
        O1 = LUT[MSB0][MSB1][MSB2 + 1][MSB3]
        O2 = LUT[MSB0][MSB1][MSB2 + 1][MSB3 + 1]
        O3 = LUT[MSB0][MSB1 + 1][MSB2 + 1][MSB3 + 1]
    else:
        w0 = W - LSB3
        w1 = LSB3 - LSB2
        w2 = LSB2 - LSB1
        w3 = LSB1 - LSB0
        w4 = LSB0
        O1 = LUT[MSB0][MSB1][MSB2][MSB3 + 1]
        O2 = LUT[MSB0][MSB1][MSB2 + 1][MSB3 + 1]
        O3 = LUT[MSB0][MSB1 + 1][MSB2 + 1][MSB3 + 1]

    O0 = LUT[MSB0][MSB1][MSB2][MSB3]
    O4 = LUT[MSB0 + 1][MSB1 + 1][MSB2 + 1][MSB3 + 1]
    return (w0 * O0 + w1 * O1 + w2 * O2 + w3 * O3 + w4 * O4) / W 

def PSNR_LUT_test(img1, img2):
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return 100
    pixel_max = 255.0
    return 20 * math.log10(pixel_max / math.sqrt(mse))



if __name__ == "__main__":
    LUT = np.load("LUT.npy")
    lr_path = "dataset/Set14/LRbicx4"
    hr_path = "dataset/Set14/original"
    file_name_list = os.listdir(lr_path)
    W = 16  
    psnr = 0
    

    dir_name = "LUT_test"
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    for file in file_name_list:

        test_img = np.array(Image.open(lr_path + "/" + file))
        # print(test_img.shape)   hwc
        test_img = np.transpose(test_img, (2, 0, 1)) # hwc -> chw
        test_img = np.pad(test_img, ((0, 0), (0, 1), [0, 1]), 'reflect')
        



        c = test_img.shape[0]
        h = test_img[0].shape[0]
        w = test_img[0].shape[1]
        sr = np.zeros((3, 4 * (h - 1), 4 * (w - 1)))
        for i in range(c):
            for j in range(h - 1):
                for k in range(w - 1):
                    RF = test_img[i][j : j+2, k : k+2]
                    I0 = RF[0][0]
                    I1 = RF[0][1]
                    I2 = RF[1][0]
                    I3 = RF[1][1]
                    sr_patch = tetrahedral_interpolation(LUT, I0, I1, I2, I3, W)

                    sr[i][4 * j : 4 * j + 4, 4 * k : 4 * k + 4] = sr_patch

        hr_img = np.array(Image.open(hr_path + "/" + file))
        hr_img = np.transpose(hr_img, (2, 0, 1))
        hr_img = np.resize(hr_img, (c, 4 * (h - 1), 4 * (w - 1)))
        psnr = psnr + PSNR_LUT_test(np.uint8(np.clip(sr, 0, 255)), hr_img)

        img = np.clip(sr, 0, 255)
        # print(sr[number])
        img = img.transpose(1, 2, 0)  # chw -> hwc
        img = Image.fromarray(np.uint8(img)).convert('RGB')

        img.save(dir_name + '/' + file)

    print("psnr:", str(psnr / 14.))