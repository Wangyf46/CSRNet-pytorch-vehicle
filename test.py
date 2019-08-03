import os
import glob
import time
import numpy as np
import torch
import cv2
import h5py
import argparse
import scipy.io as io
import matplotlib.pylab as plt
from PIL import Image
import ipdb

from src.model import CSRNet
from src.utils import *


parser = argparse.ArgumentParser(description = "PyTorch CSRNet vehicle Test")
parser.add_argument("--gpu", default = '3', type = str, 
                    help = "GPU id to use.")
parser.add_argument("--output_dir",
                    default = "/data/wangyf/Output/CSRNet_vehicle/", 
                    type = str) 

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

start = time.time()
output_dir = args.output_dir + "s_1_100_0320model_best/"  


start = time.time()
root = "/data/wangyf/datasets/TRANCOS_v3/"
file_results = output_dir + "test_result.txt"
test_list = []   # 421

with open(root + "image_sets/test.txt", 'r') as f:
    for line in f.readlines():
        line = line.strip("\n")
        test_list.append(line)
model = CSRNet()
checkpoint = torch.load("s_1_100_0320model_best.pth.tar")
model.load_state_dict(checkpoint['state_dict'])
model = model.cuda().eval()
mae = 0.0
mse = 0.0
f = open(file_results, 'w')
for img_path in test_list:
    img_path = root + "images/" + img_path
    print(img_path)
    label_count = 0.0
    gt_path = os.path.splitext(img_path)[0] + ".h5"
    mask_path = os.path.splitext(img_path)[0] + "mask.mat"
    img = cv2.imread(img_path)                               # RGB mode, (w,h)
    img_mask = io.loadmat(mask_path)["BW"]
    (R, G, B) = cv2.split(img) * img_mask
    img = cv2.merge([R, G, B])
    img = Image.fromarray(img)

    gt_file = h5py.File(gt_path)
    gt_density_map = np.asarray(gt_file['density'])

    gt_count = np.sum(gt_density_map)
    et_density_map = et_data(img, model)
    et_count = et_density_map.sum()
    
    print(et_count, gt_count)
    ipdb.set_trace()
    ## accuracy and robustness
    diff = gt_count - et_count
    mae += abs(diff)
    mse += diff ** 2

    ## data normalization
    gt_density_map = gt_density_map * 255 / np.max(gt_density_map)
    et_density_map = et_density_map * 255 / np.max(et_density_map)

    ## quality assessment
    PSNR, gt_density_map_interpolation = get_quality_psnr(gt_density_map, 
                                                          et_density_map)
    SSIM = get_quality_ssim(gt_density_map_interpolation, 
                                  et_density_map)

    ## visualization and save
    path = img_path.replace("/data/wangyf/datasets/TRANCOS_v3/images/", "")  ## todo
    print(path)
    save_path = output_dir + "img_gt_et_" + path
    save_path_interpolation = output_dir + "img_gt_et_interpolation_" + path
    
    visualization_save(save_path, img, gt_density_map, et_density_map,
                             label_count, gt_count, et_count, PSNR, SSIM)
    visualization_save(save_path_interpolation, img, 
                             gt_density_map_interpolation, 
                             et_density_map,
                             label_count, gt_count, et_count, PSNR, SSIM)
 


    '''
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.axis("off")
    plt.subplot(1,3,2)
    plt.imshow(gt_density_map, cmap = plt.cm.jet)
    plt.axis("off")
    plt.subplot(1,3,3)
    plt.imshow(et_density_map, cmap = plt.cm.jet)
    plt.axis("off")
    #plt.savefig(path, bbox_inches = "tight")  # remove blank, only can save and cannot display
    plt.show()
    plt.close()
    '''

    f.write("Image_{path};  "
            "label_count:{label_count:.2f}"
            "gt_count:{gt_count:.2f};  "
            "et_count:{et_count:.2f};  "
            "diff:{diff:.2f}  "
            "PSNR:{PSNR:.2f}; "
            "SSIM:{SSIM:.2f};\n "
            .format(path = path, 
                    label_count = label_count, 
                    gt_count = gt_count, 
                    et_count = et_count,
                    diff = diff, 
                    PSNR = PSNR, 
                    SSIM = SSIM))
end = time.time()
second = end - start
print("elapsed time:{elapsed_time:.4f};  "
      "fps:{fps:.4f};   "
      "mae:{mae:.4f};   "
      "mse:{mse:.4f};\n "
      .format(elapsed_time = second, 
              fps = len(test_list) / second,
              mae = mae / len(test_list), 
              mse = np.sqrt(mse / len(test_list))))
f.write("elapsed time:{elapsed_time:.4f};  "
        "fps:{fps:.4f};   "
        "mae:{mae:.4f};   "
        "mse:{mse:.4f};\n "
        .format(elapsed_time = second, 
                fps = len(test_list) / second,
                mae = mae / len(test_list), 
                mse = np.sqrt(mse / len(test_list))))
f.close()

