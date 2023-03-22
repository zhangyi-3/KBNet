
## Evaluation

1. Download the pre-trained [model](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155135732_link_cuhk_edu_hk/EofsV3eVcAxNlrW72JXqzRUBhkM1Mzw50pJ3BHlAyMYnVw?e=VguIDQ) and place it in `./pretrained_models/`

2. Download test datasets (Test1200, Test2800), run 
```
python download_data.py --data test
```

3. Testing
```
python -u test.py --yml Options/kbnet_l.yml

evaluate_PSNR_SSIM.m 
```
