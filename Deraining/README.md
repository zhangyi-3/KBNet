
## Evaluation

1. Download the pre-trained [model](https://pan.baidu.com/s/1AVo11Z0J9HKg-4YE00ej-w?pwd=m8tv) and place it in `./pretrained_models/`

2. Download test datasets (Test1200, Test2800), run 
```
python download_data.py --data test
```

3. Testing
```
python -u test.py --yml Options/kbnet_l.yml

evaluate_PSNR_SSIM.m 
```
