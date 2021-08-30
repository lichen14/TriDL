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
[![](https://github.com/lichen14/TriDL/blob/master/display/introduction.png)

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
3. Optional. To uninstall this package, run:
```bash
$ pip uninstall TriDL
```

### Datasets
* By default, the datasets are put in ```<root_dir>/dataset```.
* An alternative option is to explicitlly specify the parameters ```DATA_DIRECTORY_SOURCE``` and ```DATA_DIRECTORY_TARGET``` in YML configuration files.
* Download the [MMWHS Dataset](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mmwhs/), the MRI dataset is used as source domain dataset while the CT dataset for target domain dataset. (For participants who want to download and use the data, they need to agree with the conditions above and the terms in the registration form in above website.)
* After download, The MMWHS dataset directory should have this basic structure:
```bash
<root_dir>/dataset/mri_dataset/                               % MRI samples root
<root_dir>/dataset/mri_dataset/train/image/                   % MRI images
<root_dir>/dataset/mri_dataset/train/label/                   % MRI annotation
<root_dir>/dataset/mri_list/                                  % MRI samples list

<root_dir>/dataset/ct_dataset/                                % CT samples root
<root_dir>/dataset/ct_dataset/train/image/                    % CT images
<root_dir>/dataset/ct_dataset/train/label/                    % CT annotation
<root_dir>/dataset/ct_list/                                   % CT samples list
...
```
### Pre-trained models
* Initial pre-trained model can be downloaded from [DeepLab-V2](https://drive.google.com/open?id=1TIrTmFKqEyf3pOKniv8-53m3v9SyBK0u)
* Transferred images for MMWHS dataset can be found:
  * [MRI as CT (MMWHS)_need updation](https://drive.google.com/open?id=1OBvYVz2ND4ipdfnkhSaseT8yu2ru5n5l)
  
## Running the code
The well-trained model can be downloaded here [TriDL_deeplab](https://drive.google.com/open?id=1uNIydmPONNh29PeXqCb9MGRAnCWxAu99). You can use the pre-trained model or your own model to make a test as following:
```bash
$ cd <root_dir>/advent
$ python test.py --cfg ./configs/<your_yml_name>.yml --exp-suffix <your_define_suffix>
```
### Training adaptive segmenation network in TriDL
To ensure reproduction, the random seed has been fixed in the code. Still, you may need to train a few times to reach the comparable performance.

By default, logs and snapshots are stored in ```<root_dir>/experiments``` with this structure:
```bash
<root_dir>/experiments/logs
<root_dir>/experiments/snapshots  %output and trained model are stored in this file.
```

To train TriDL:
```bash
$ cd <root_dir>/advent
$ python train_MRI2CT.py --cfg ./configs/advent.yml
$ python train.py --cfg ./configs/<your_yml_name>.yml  --exp-suffix <your_define_suffix>  --tensorboard         % using tensorboard
```


### Well-trained models and our implemented methods will be released soon.

## Acknowledgements
This codebase is heavily borrowed from [AdaptSegNet](https://github.com/wasidennis/AdaptSegNet) and [ADVENT](https://github.com/valeoai/ADVENT).
Thanks to following repos for sharing and we referred some of their codes to construct TriDL:
### References
- [SIFA](https://github.com/cchen-cc/SIFA)
- [Attention-Gated-Networks](https://github.com/ozan-oktay/Attention-Gated-Networks)
- [SegmenTron](https://github.com/LikeLy-Journey/SegmenTron)
- [MICCAI-MMWHS-2017](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mmwhs/)
- [BDL](https://github.com/liyunsheng13/BDL)
