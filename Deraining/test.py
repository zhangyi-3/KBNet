import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
from tqdm import tqdm
from natsort import natsorted
from glob import glob
from skimage import img_as_ubyte

import utils
from basicsr.models.archs.kbnet_l_arch import KBNet_l

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

parser = argparse.ArgumentParser(description='Image Deraining using Restormer')

parser.add_argument('--input_dir', default='./Datasets/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--yml', default='Options/kbnet_l.yml', type=str)
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

factor = 8
datasets = ['Test1200', 'Test2800']

for dataset in datasets:
    result_dir = os.path.join(args.result_dir, dataset)
    os.makedirs(result_dir, exist_ok=True)

    inp_dir = os.path.join(args.input_dir, 'test', dataset, 'input')
    files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
    with torch.no_grad():
        for file_ in tqdm(files):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = np.float32(utils.load_img(file_)) / 255.
            img = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img.unsqueeze(0).cuda()

            # Padding in case images are not multiples of 8
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            restored = model_restoration(input_)

            # Unpad images to original dimensions
            restored = restored[:, :, :h, :w]

            restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            save_name = os.path.join(result_dir,
                                     os.path.splitext(os.path.split(file_)[-1])[0] + '-' + name.split('-')[-1] + '.png')
            utils.save_img(save_name, img_as_ubyte(restored))
