import os
import argparse
import lpips
import yaml

import torch
import torch.nn as nn
import numpy as np

from natsort import natsorted
from glob import glob
from tqdm import tqdm

import utils
from basicsr.models.archs.kbnet_l_arch import KBNet_l
from basicsr.utils.util import patch_forward

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

alex = lpips.LPIPS(net='alex').cuda()

parser = argparse.ArgumentParser(description='Single Image Defocus Deblurring using Restormer')

parser.add_argument('--input_dir', default='./Datasets/test/DPDD/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/Single_Defocus/', type=str, help='Directory for results')
parser.add_argument('--yml', default='Options/kbnet_l.yml', type=str)
parser.add_argument('--save_images', default=False, help='Save denoised images in result directory')

args = parser.parse_args()

####### Load yaml #######

yaml_file = args.yml
name = os.path.basename(yaml_file).split('.')[0]

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
pth_path = x['path']['pretrain_network_g']
print('**', yaml_file, pth_path)

##########################

model_restoration = eval(s)(**x['network_g'])

checkpoint = torch.load(pth_path)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ", pth_path)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

result_dir = args.result_dir
if args.save_images:
    os.makedirs(result_dir, exist_ok=True)

filesI = natsorted(glob(os.path.join(args.input_dir, 'inputC', '*.png')))
filesC = natsorted(glob(os.path.join(args.input_dir, 'target', '*.png')))

indoor_labels = np.load('./Datasets/test/DPDD/indoor_labels.npy')
outdoor_labels = np.load('./Datasets/test/DPDD/outdoor_labels.npy')

psnr, mae, ssim, pips = [], [], [], []
with torch.no_grad():
    for fileI, fileC in tqdm(zip(filesI, filesC), total=len(filesC)):

        imgI = np.float32(utils.load_img(fileI)) / 255.
        imgC = np.float32(utils.load_img(fileC)) / 255.

        patchI = torch.from_numpy(imgI).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        patchC = torch.from_numpy(imgC).unsqueeze(0).permute(0, 3, 1, 2).cuda()

        # restored = model_restoration(patchI)
        restored = patch_forward(patchI, model_restoration, skip=1024, padding=64)

        restored = torch.clamp(restored, 0, 1)
        pips.append(alex(patchC, restored, normalize=True).item())

        restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        psnr.append(utils.PSNR(imgC, restored))
        mae.append(utils.MAE(imgC, restored))
        ssim.append(utils.SSIM(imgC, restored))
        if args.save_images:
            save_file = os.path.join(result_dir, os.path.split(fileC)[-1])
            save_file = save_file[:-4] + '-%s.png' % (name)
            restored = np.uint8((restored * 255).round())
            utils.save_img(save_file, restored)

psnr, mae, ssim, pips = np.array(psnr), np.array(mae), np.array(ssim), np.array(pips)

psnr_indoor, mae_indoor, ssim_indoor, pips_indoor = psnr[indoor_labels - 1], mae[indoor_labels - 1], ssim[
    indoor_labels - 1], pips[indoor_labels - 1]
psnr_outdoor, mae_outdoor, ssim_outdoor, pips_outdoor = psnr[outdoor_labels - 1], mae[outdoor_labels - 1], ssim[
    outdoor_labels - 1], pips[outdoor_labels - 1]

print("Indoor:  PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f}".format(np.mean(psnr_indoor), np.mean(ssim_indoor),
                                                                    np.mean(mae_indoor), np.mean(pips_indoor)))
print("Outdoor: PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f}".format(np.mean(psnr_outdoor), np.mean(ssim_outdoor),
                                                                    np.mean(mae_outdoor), np.mean(pips_outdoor)))
