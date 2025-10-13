

## Evaluation

- Download the pre-trained [models](https://pan.baidu.com/s/1PbpDoYSlsoCJvVHvfFngKQ?pwd=gyh4) and place them in `./pretrained_models/`

- Download test dataset, run
```
python download_data.py --data test
```

- Testing on **single-image** defocus deblurring task, run
```
python test.py --yml Options/kbnet_l.yml
```
