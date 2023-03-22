%% Restormer: Efficient Transformer for High-Resolution Image Restoration
%% Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
%% https://arxiv.org/abs/2111.09881


clc;close all;clear all;

% datasets = {'Rain100L'};
datasets = {'Test2800', 'Test1200'};
num_set = length(datasets);

psnr_alldatasets = 0;
ssim_alldatasets = 0;

tic
delete(gcp('nocreate'))
parpool('local', 4);

for idx_set = 1:num_set
    file_path = strcat('./results/', datasets{idx_set}, '/');
    gt_path = strcat('./Datasets/test/', datasets{idx_set}, '/target/');
    path_list = [dir(strcat(file_path,'*.jpg')); dir(strcat(file_path,'*.png'))];
    gt_list = [dir(strcat(gt_path,'*.jpg')); dir(strcat(gt_path,'*.png'))];
    img_num = length(path_list);

    total_psnr = 0;
    total_ssim = 0;
    if img_num > 0 
        parfor j = 1:img_num
           image_name = path_list(j).name;
           gt_name = gt_list(j).name;
           input = imread(strcat(file_path,image_name));
           gt = imread(strcat(gt_path, gt_name));
           ssim_val = compute_ssim(input, gt);
           psnr_val = compute_psnr(input, gt);
           total_ssim = total_ssim + ssim_val;
           total_psnr = total_psnr + psnr_val;
       end
    end
    qm_psnr = total_psnr / img_num;
    qm_ssim = total_ssim / img_num;

    fprintf('For %s dataset PSNR: %f SSIM: %f\n', datasets{idx_set}, qm_psnr, qm_ssim);

    psnr_alldatasets = psnr_alldatasets + qm_psnr;
    ssim_alldatasets = ssim_alldatasets + qm_ssim;
    
end

fprintf('For all datasets PSNR: %f SSIM: %f\n', psnr_alldatasets/num_set, ssim_alldatasets/num_set);

delete(gcp('nocreate'))
toc





