import os
import shutil
import yaml
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import utils_tool

from basicsr.models.archs.kbnet_s_arch import KBNet_s
from basicsr.utils.util import patch_forward

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

parser = argparse.ArgumentParser(description='Gaussian Color Denoising using Restormer')

parser.add_argument('--input_dir', default='./Datasets/test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/Gaussian_Color_Denoising/', type=str,
                    help='Directory for results')
parser.add_argument('--yml', default='Options/gau_color_50.yml', type=str)
args = parser.parse_args()

factor = 8

yaml_file = args.yml
x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

sigmas = [x['datasets']['train']['sigma_range']]
cfg_name = os.path.basename(yaml_file).split('.')[0]
pth_path = x['path']['pretrain_network_g']
print('**', yaml_file, pth_path)

s = x['network_g'].pop('type')
model_restoration = eval(s)(**x['network_g'])
checkpoint = torch.load(pth_path)

print("===>Testing using weights: ")
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.load_state_dict(checkpoint['net'])
model_restoration.eval()

datasets = ['Kodak', 'CBSD68', 'McMaster', 'Urban100']

for sigma_test in sigmas:
    print("Compute results for noise level", sigma_test)

    for dataset in datasets:
        inp_dir = os.path.join(args.input_dir, dataset)
        files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.tif')))
        result_dir_tmp = os.path.join(args.result_dir, 'color')
        os.makedirs(result_dir_tmp, exist_ok=True)

        psnr_list = []

        with torch.no_grad():
            for file_ in tqdm(files):
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
                gt = utils_tool.load_img(file_)

                img = np.float32(gt) / 255.

                np.random.seed(seed=0)  # for reproducibility
                img += np.random.normal(0, sigma_test / 255., img.shape)
                noisy = torch.from_numpy(img)

                img = noisy.permute(2, 0, 1)
                input_ = img.unsqueeze(0).cuda()

                # Padding in case images are not multiples of 8
                h, w = input_.shape[2], input_.shape[3]
                H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
                padh = H - h if h % factor != 0 else 0
                padw = W - w if w % factor != 0 else 0
                input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

                # restored = patch_forward(input_, model_restoration)
                restored = model_restoration(input_)

                # Unpad images to original dimensions
                restored = restored[:, :, :h, :w]

                restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

                # save_file = os.path.join(result_dir_tmp, os.path.split(file_)[-1])
                # utils_tool.save_img(save_file, img_as_ubyte(restored))

                psnr_list.append(utils_tool.calculate_psnr(gt, img_as_ubyte(restored)))

                save_file = os.path.join(result_dir_tmp,
                                         dataset + '-' + str(sigma_test) + '-%.2f-%s-' % (psnr_list[-1], cfg_name) +
                                         os.path.split(file_)[-1])
                save_file = save_file.replace('.tif', '.jpg')
                utils_tool.save_jpg(save_file.replace('.png', '.jpg'), img_as_ubyte(restored))

                gtnoisy_path = os.path.join(args.result_dir, 'color_ori')
                os.makedirs(gtnoisy_path, exist_ok=True)
                shutil.copyfile(file_, os.path.join(gtnoisy_path, dataset + os.path.basename(file_)))
                utils_tool.save_img(
                    os.path.join(gtnoisy_path, dataset + os.path.basename(file_).split('.')[0] + '-noisy.png'),
                    img_as_ubyte(noisy.clip(0, 1)))

        print(dataset, cfg_name, np.mean(psnr_list))
