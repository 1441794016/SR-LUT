import os
import glob
import h5py
import numpy as np
import imageio

dataset_dir = "Set14/"
dataset_type = "test"

f = h5py.File("Set14_{}.h5".format(dataset_type), "w")
dt = h5py.special_dtype(vlen=np.dtype('uint8'))

for subdir in ["original", "LRbicx2", "LRbicx3", "LRbicx4"]:
    im_paths = glob.glob(os.path.join(dataset_dir, subdir, "*.png"))

    im_paths.sort()
    grp = f.create_group(subdir)

    for i, path in enumerate(im_paths):
        im = imageio.imread(path)
        print(str(i))
        print(path)
        grp.create_dataset(str(i), data=im)
