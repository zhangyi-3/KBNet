

## Evaluation

- Download the pre-trained [models](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155135732_link_cuhk_edu_hk/EofsV3eVcAxNlrW72JXqzRUBhkM1Mzw50pJ3BHlAyMYnVw?e=VguIDQ) and place them in `./pretrained_models/`

- Download test dataset, run
```
python download_data.py --data test
```

- Testing on **single-image** defocus deblurring task, run
```
python test.py --yml Options/kbnet_l.yml
```
