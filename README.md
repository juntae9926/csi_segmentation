## CSI Segmentation

We applied human segmentation task with csi dataset.

This is [VisTR paper](https://arxiv.org/abs/2011.14503)


### Dependencies
```
conda install pytorch==1.6.0 torchvision==0.7.0
```
Install pycocotools
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install git+https://github.com/youtubevos/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
```
Compile DCN module(requires GCC>=5.3, cuda>=10.0)
```
cd models/dcn
python setup.py build_ext --inplace
```

### Preparation

If you want to use pretrained DETR models [Google Drive](https://drive.google.com/drive/folders/1DlN8uWHT2WaKruarGW2_XChhpZeI9MFG?usp=sharing) [BaiduYun](https://pan.baidu.com/s/12omUNDRjhAeGZ5olqQPpHA)(passcode:alge) on COCO and save it to the pretrained path.


### Training

Training of the model requires at least 32g memory GPU, we performed the experiment on 32g V100 card. （As the training resolution is limited by the GPU memory, if you have a larger memory GPU and want to perform the experiment, please contact with me, thanks very much)

To train baseline VisTR on a single node with 8 gpus for 18 epochs, run:
```
main.py --backbone resnet50 --ytvos_path /path/to/ytvos --masks --pretrained_weights /path/to/pretrained_path
```

### Inference

```
python inference.py --masks --model_path /path/to/model_weights --save_path /path/to/results.json
```

### License

VisTR is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.

