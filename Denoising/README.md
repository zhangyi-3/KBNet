# Image Denoising
- [Gaussian Image Denoising](#gaussian-image-denoising)

[//]: # (  * [Training]&#40;#training&#41;)
  * [Evaluation](#evaluation)
      - [Grayscale blind image denoising testing](#grayscale-blind-image-denoising-testing)
      - [Grayscale non-blind image denoising testing](#grayscale-non-blind-image-denoising-testing)
      - [Color blind image denoising testing](#color-blind-image-denoising-testing)
      - [Color non-blind image denoising testing](#color-non-blind-image-denoising-testing)
- [Real Image Denoising](#real-image-denoising)

[//]: # (  * [Training]&#40;#training-1&#41;)
  * [Evaluation](#evaluation-1)
      - [Testing on SIDD dataset](#testing-on-sidd-dataset)
      - [Testing on SenseNoise dataset](#testing-on-dnd-dataset)

# Gaussian Image Denoising

## Evaluation

- Download the pre-trained [models](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155135732_link_cuhk_edu_hk/EofsV3eVcAxNlrW72JXqzRUBhkM1Mzw50pJ3BHlAyMYnVw?e=MeMB5H) and place them in `./pretrained_models/`

- Download testsets (Set12, BSD68, CBSD68, Kodak, McMaster, Urban100), run 
```
python download_data.py --data test --noise gaussian
```

#### Grayscale image denoising testing (sigma=25)

- To obtain denoised predictions, run
```
python -u test_gaussian_gray_denoising.py --yml Options/gau_gray_25.yml
```

#### Color blind image denoising testing (sigma=50)

```
python -u test_gaussian_color_denoising.py --yml Options/gau_color_50.yml
```


<hr />

# Real Image Denoising

## Evaluation

- Download the pre-trained [models](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155135732_link_cuhk_edu_hk/EofsV3eVcAxNlrW72JXqzRUBhkM1Mzw50pJ3BHlAyMYnVw?e=MeMB5H) and place them in `./pretrained_models/`

#### Testing on SIDD dataset

- Download SIDD validation data, run 
```
python download_data.py --noise real --data test --dataset SIDD
```

- To obtain denoised results, run
```
python -u test_real_denoising_sidd.py --yml Options/sidd.yml
```

[//]: # (- To reproduce PSNR/SSIM scores on SIDD data, run)

[//]: # (```)

[//]: # (evaluate_sidd.m)

[//]: # (```)

#### Testing on [SenseNoise dataset](https://github.com/zhangyi-3/IDR)

- Download the SenseNoise [testing data](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155135732_link_cuhk_edu_hk/Eqj2xo-jzTlChyuyF-JWmzQBdi5sLBrRZxQikdnko3EpfQ?e=81pFjp) and place them in `./Datasets/`

- To obtain denoised results, run
```
python test_real_denoising_sense500.py --yml Options/sensenoise.yml
```
