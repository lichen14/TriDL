# TriDL-Pytorch
Official Pytorch Implementation for TriDL.

## Updates
- *08/2021*: Check out our paper [Tri-Directional Tasks Complementary Learning for Unsupervised Domain Adaptation of Cross-modality Medical Image Semantic Segmentation] (submitted to BIBM 2021). With the Tri-directional learning framework, we synergize image style transformation, mask segmentation and
edge segmentation, and mutually boost every task through complementary training in each iteration. this work was proposed to solve domain shift in the task of medical image semantic segmentation.

## Paper
![](https://github.com/lichen14/TriDL/blob/master/display/framework.png)

If you find this code useful for your research, please cite our [paper](https://arxiv.org):

```
@inproceedings{li2021tridl,
  title={Tri-Directional Tasks Complementary Learning for Unsupervised Domain Adaptation of Cross-modality Medical Image Semantic Segmentation},
  coming soon.
}
```
## Demo
![](https://github.com/lichen14/TriDL/blob/master/display/introduction.png)

## Preparation
### Requirements

- Hardware: PC with NVIDIA 1080T GPU. (others are alternative.)
- Software: *Ubuntu 18.04*, *CUDA 10.0.130*, *pytorch 1.3.0*, *Python 3.6.9*
- Package:
  - `torchvision`
  - `tensorboardX`
  - `scikit-learn`
  - `glob`
  - `matplotlib`
  - `skimage`
  - `medpy`
  - `tqdm`
### Installation
0. Clone the repo:
```bash
$ git clone https://github.com/lichen14/TriDL
$ cd TriDL
```
1. Install OpenCV if you don't already have it:

```bash
$ conda install -c menpo opencv
```
2. Install this repository and the dependencies using pip:
```bash
$ pip install -e <root_dir>
```
With this, you can edit the TriDL code on the fly and import function 
and classes of TriDL in other project as well.
3. Install [PyTorch & torchvision](http://pytorch.org/)
Follow the instructions in [pytorch.org](http://pytorch.org) for your current setup

4. Install [Visdom](https://github.com/facebookresearch/visdom)
To plot loss graphs and draw images in a nice web browser view
```bash
$ pip install visdom
```
5. Optional. To uninstall this package, run:
```bash
$ pip uninstall TriDL
```

### Datasets
* By default, the datasets are put in ```<root_dir>/dataset```.
* An alternative option is to explicitlly specify the parameters ```DATA_DIRECTORY_SOURCE``` and ```DATA_DIRECTORY_TARGET``` in YML configuration files.
* Download the [MMWHS Dataset](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mmwhs/), the MRI dataset is used as source domain dataset while the CT dataset for target domain dataset. (For participants who want to download and use the data, they need to agree with the conditions above and the terms in the registration form in above website.)
* After download, The MMWHS dataset directory should have this basic structure:
```
    .
    ├── datasets                   
    |   ├── <dataset_name>         # i.e. MRI2CT
    |   |   ├── train              # Training
    |   |   |   ├── A              # Contains domain A images (i.e. MRI)
    |   |   |   └── B              # Contains domain B images (i.e. CT)
    |   |   └── test               # Testing
    |   |   |   ├── A              # Contains domain A images (i.e. MRI)
    |   |   |   └── B              # Contains domain B images (i.e. CT)
 ```

### Pre-trained models and translated images 
* Initial pre-trained model can be downloaded from [DeepLab-V2](https://drive.google.com/open?id=1TIrTmFKqEyf3pOKniv8-53m3v9SyBK0u)
  
## Evaluation
* The well-trained model can be downloaded here [TriDL_deeplab](https://pan.baidu.com/s/1LUNAVwJXp8T0PPceG2QnMg)（password:wcn8）. 
* You can use the pre-trained model or your own model to make a test as following:
```bash
$ cd <root_dir>/tridl
$ python test.py --cfg ./configs/<your_yml_name>.yml --exp-suffix <your_define_suffix>
```
## Training

### Training the cross-modality style translator in TriDL
* Before training, you need open the visdom and tensorboard for visualization.
```bash
python -m visdom.server
tensorboard --logdir=<your_tfb_log_dir>
```
* You can also view the training progress as well as live output images by running ```python3 -m visdom``` in another terminal and opening [http://localhost:8097/](http://localhost:8097/) in your favourite web browser. This would show training loss progress and translated images.

![](https://github.com/lichen14/TriDL/blob/master/display/visdom.png)

* Following command will start a training session using the images under the *dataroot/train* directory with the hyperparameters that showed best results according to CycleGAN authors. You are free to change those hyperparameters, see ```python train_tfb.py --help``` for a description of those.
```
cd <root_dir>/PyTorch-CycleGAN-cleaner/PyTorch-CycleGAN-master
python train_tfb.py --dataroot datasets/<dataset_name>/ --cuda --name <your_name> --batchsize N
```
* Both generators and discriminators weights will be saved under the output directory.
* If you don't own a GPU remove the --cuda option, although I advise you to get one!
* Taking my command as an example:
```
python train_tfb.py --dataroot /home/lc/Study/Project/PyTorch-CycleGAN-cleaner/PyTorch-CycleGAN-master/datasets/MRI2CT/ --cuda --name <MRI2CT-UDA-1> --batchsize 4
```
### Testing the translator in TriDL
* Following command will take the images under the *dataroot/test* directory, run them through the generators and save the output under the *output/A* and *output/B* directories. As with train, some parameters like the weights to load, can be tweaked, see ```python test.py --help``` for more information.
```
python test.py --dataroot datasets/<dataset_name>/ --cuda  --name <your_name> --batchsize N
```
* Translated images for MMWHS dataset can be found:
  * [CT slice (from MMWHS)](https://pan.baidu.com/s/1du8Tayjrr_IK3YCzDgiQ_Q)(password:9133).
  * We only did the MRI to CT domain adaptation in this work, so there is only translated CT images uploaded, the translated MRI images will uploaded in the further work.
  
### Training adaptive segmenation networks in TriDL
* before training, the translated dataset has the following structure:
```bash
<root_dir>/tridl/dataset/mri_dataset/                               % MRI samples root
<root_dir>/tridl/dataset/mri_dataset/train/image/                   % MRI images and translated images 
<root_dir>/tridl/dataset/mri_dataset/train/label/                   % MRI annotation
<root_dir>/tridl/dataset/mri_list/                                  % MRI samples list

<root_dir>/tridl/dataset/ct_dataset/                                % CT samples root
<root_dir>/tridl/dataset/ct_dataset/train/image/                    % CT images and translated images 
<root_dir>/tridl/dataset/ct_dataset/train/label/                    % CT annotation
<root_dir>/tridl/dataset/ct_list/                                   % CT samples list
...
```
* To ensure reproduction, the random seed has been fixed in the code. Still, you may need to train a few times to reach the comparable performance.
* By default, logs and snapshots are stored in ```<root_dir>/experiments``` with this structure:
```bash
<root_dir>/experiments/logs
<root_dir>/experiments/snapshots  %output and trained model are stored in this file.
```
* To train TriDL:
```bash
$ cd <root_dir>/tridl
$ python train_MRI2CT.py --cfg ./configs/advent.yml
$ python train.py --cfg ./configs/<your_yml_name>.yml  --exp-suffix <your_define_suffix>  --tensorboard         % using tensorboard
```


### Well-trained models and our implemented methods can be downloaded in the links.
* the well-trained TriDL [model](https://pan.baidu.com/s/1LUNAVwJXp8T0PPceG2QnMg)（password:wcn8）(after two round tri-directional complementary learning).
* the half-trained TriDL [model](https://pan.baidu.com/s/1ZYb9TyTrr6C81N3WGW55xA)（password:5ndg）(after one round tri-directional complementary learning).
* the supervised-trained [model](https://pan.baidu.com/s/19g9i_Qqwc7URY0pFBgSboA)（password:i76n）(use annotated CT images and labels to train segmentor and evaluate on the CT images).
* the non-adaptation-trained [model](https://pan.baidu.com/s/1yRzrASgXk2qnw-vOtdHjDw)（password:1w1n）(use annotated MRI images and labels to train segmentor and evaluate on the CT images).
* the BDL [model](https://pan.baidu.com/s/14CSfvz-bJNS1AwpATwadaw)（password:83qk）(use annotated MRI images and labels to train segmentor and evaluate on the CT images).

## Acknowledgements
This codebase is heavily borrowed from [SIFA](https://github.com/cchen-cc/SIFA) and [cyclegan](https://github.com/aitorzip/PyTorch-CycleGAN).
Thanks to following repos for sharing and we referred some of their codes to construct TriDL:
### References
- [Attention-Gated-Networks](https://github.com/ozan-oktay/Attention-Gated-Networks)
- [SegmenTron](https://github.com/LikeLy-Journey/SegmenTron)
- [MICCAI-MMWHS-2017](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mmwhs/)
- [BDL](https://github.com/liyunsheng13/BDL)
