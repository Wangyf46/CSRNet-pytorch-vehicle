from torch.utils.data import Dataset
import random
from PIL import Image
import numpy as np
import h5py
import cv2
import os
from matplotlib import pyplot as plt
import scipy.io as io

path = "/data/wangyf/datasets/TRANCOS_v3/images/"

class listDataset(Dataset):
    def __init__(self, root, shape = None, shuffle = True, transform = None,
                 train = False, seen = 0, batch_size = 1, num_workers = 4):
        if train:
            root = 4 * root
        random.shuffle(root)
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen                    # 0?
        self.batch_size = batch_size
        self.num_workers = num_workers      # 4?

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        img_path = path + self.lines[index]
        mask_path = os.path.splitext(img_path)[0] + "mask.mat"
        gt_path = os.path.splitext(img_path)[0] + ".h5"
        img = cv2.imread(img_path)                   # RGB mode, (w,h)
        img_mask = io.loadmat(mask_path)["BW"]
        (R, G, B) = cv2.split(img) * img_mask
        img = cv2.merge([R, G, B])
        img = Image.fromarray(img)

        gt_file = h5py.File(gt_path)
        target = np.asarray(gt_file['density'])
        ## pooling effect
        shape1 = int(target.shape[1] / 8.0)                         # w
        shape0 = int(target.shape[0] / 8.0)                         # h
        target = cv2.resize(target, (shape1, shape0)) * 64       # (h/8, w/8)
        if self.transform is not None:
            img = self.transform(img)                               # torch.Size([3,h,w])
        return img, target