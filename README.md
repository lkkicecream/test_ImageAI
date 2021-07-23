# test_ImageAI

## Library
[ImageAI](https://github.com/OlafenwaMoses/ImageAI0)

## Dependies
- Anaconda
- Python 3.7.6
- pip
- Tensorflow 2.4.0
- keras 2.4.3 
- numpy 1.19.3 
- pillow 7.0.0 
- scipy 1.4.1 
- h5py 2.10.0 
- matplotlib 3.3.2 
- opencv-python 
- keras-resnet 0.2.0

## execution_path
- put images in  a new folder "night_images"
- output will put in "retest"

### 想法
- 透過 ImageAI 框出車輛
- 以遮罩留下感興趣區域（ROI）
- 確認遮罩內是否有初被框的車輛
- 以此判斷車前包含車輛、車距過近


## 研究心得
- 白天偵測準確，夜晚不佳
- 使用 yolo3.h5 配合 speed=flash 是最快的
