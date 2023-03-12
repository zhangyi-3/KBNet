import os
import shutil
import yaml
import argparse

import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm
from skimage import img_as_ubyte

import utils_tool
from basicsr.utils.util import patch_forward
from basicsr.models.archs.kbnet_s_arch import KBNet_s
from basicsr.models.archs.kbnet_l_arch import KBNet_l

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

parser = argparse.ArgumentParser(description='Gaussian Color Denoising using Restormer')

parser.add_argument('--input_dir', default='./Datasets/test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str,
                    help='Directory for results')
parser.add_argument('--yml', default=None, type=str, help='Directory for results')

args = parser.parse_args()

yaml_file = args.yml
x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

pth_path = x['path']['pretrain_network_g']
cfg_name = os.path.basename(args.yml).split('.')[0]

s = x['network_g'].pop('type')
model_restoration = eval(s)(**x['network_g'])

checkpoint = torch.load(pth_path)
model_restoration.load_state_dict(checkpoint['model'])
print("===>Testing using weights: ")
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

factor = 8
skip, padding = 1024, 64

inp_dir = './Datasets/sensenoise/final_datasetv4_png/test.txt'
noisy_path = './Datasets/sensenoise/final_datasetv4_png/noisy'

files = np.loadtxt(inp_dir, dtype=np.str)
print('** len file', len(files))
result_dir_tmp = os.path.join(args.result_dir, 'sensenoise', cfg_name)
os.makedirs(result_dir_tmp, exist_ok=True)

psnr_list = []
ssim_list = []

with torch.no_grad():
    for file_ in tqdm(files):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        file_ = os.path.join(noisy_path, file_.replace('gt', 'noisy'))
        img = utils_tool.load_img(file_)
        img = np.float32(img) / 255.

        gt = utils_tool.load_img(file_.replace('noisy', 'gt'))
        gtnoisy_path = os.path.join(args.result_dir, 'sensenoise_ori')
        os.makedirs(gtnoisy_path, exist_ok=True)
        shutil.copyfile(file_, os.path.join(gtnoisy_path, os.path.basename(file_)))
        shutil.copyfile(file_.replace('noisy', 'gt'),
                        os.path.join(gtnoisy_path, os.path.basename(file_.replace('noisy', 'gt'))))

        img = torch.from_numpy(img).permute(2, 0, 1)
        input_ = img.unsqueeze(0).cuda()

        # split to patch inference
        restored = patch_forward(input_, model_restoration, skip=skip, padding=padding)

        restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        _psnr = utils_tool.calculate_psnr(gt, img_as_ubyte(restored))
        psnr_list.append(_psnr)
        ssim_list.append(utils_tool.calculate_ssim(gt, img_as_ubyte(restored)))

        # save_file = os.path.join(result_dir_tmp, os.path.split(file_)[-1])
        # utils_tool.save_img(save_file, img_as_ubyte(restored))
        save_file = os.path.join(result_dir_tmp, os.path.split(file_)[-1][:-4] + '-%.2f-%s.jpg' % (_psnr, cfg_name))
        # utils_tool.save_jpg(save_file, img_as_ubyte(restored))

print('sensenoise', cfg_name, np.mean(psnr_list), 'ssim', np.mean(ssim_list))
