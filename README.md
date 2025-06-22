# 4K4DGen: Panoramic 4D Generation at 4K Resolution


<p align="center">
<a href="https://arxiv.org/abs/2406.13527"><img src="https://img.shields.io/badge/Arxiv-2406.13527-B31B1B.svg"></a>
<a href="https://4k4dgen.github.io/"><img src="https://img.shields.io/badge/Project-Page-048C3D"></a>
<a href="https://github.com/ShadowIterator/4K4DGen"><img src="https://img.shields.io/github/stars/ShadowIterator/4K4DGen"></a>
</p>


<div>
    Renjie Li<sup>1,4</sup>&nbsp;&nbsp;&nbsp;
    Panwang Pan<sup>1</sup>&nbsp;&nbsp;&nbsp;
    Bangbang Yang<sup>1</sup>&nbsp;&nbsp;&nbsp;
    Dejia Xu<sup>2</sup>&nbsp;&nbsp;&nbsp;
    Shijie Zhou<sup>3</sup>&nbsp;&nbsp;&nbsp;
    Xuanyang Zhang<sup>1</sup>&nbsp;&nbsp;&nbsp;
    Zeming Li<sup>1</sup>&nbsp;&nbsp;&nbsp;
    Achuta Kadambi<sup>3</sup>&nbsp;&nbsp;&nbsp;
    Zhangyang Wang<sup>2</sup>&nbsp;&nbsp;&nbsp;
    Zhengzhong Tu<sup>4</sup>&nbsp;&nbsp;&nbsp;
    Zhiwen Fan<sup>2</sup>&nbsp;&nbsp;&nbsp;
</div>

<div>
    <span><sup>1</sup>ByteDance </span>
    <span>&nbsp;&nbsp;</span>
    <span><sup>2</sup>UT Austin </span>
    <span>&nbsp;&nbsp;</span>
    <span><sup>3</sup>UCLA </span>
    <span>&nbsp;&nbsp;</span>
    <span><sup>4</sup>TAMU </span>
</div>



-----------------------------

![HeadDemo](output/example/I2.gif)

## Release Checklist
- [x] Complete documents
- [x] Code release for Animation Phase
- [x] Code release for Lifting Phase
- [x] Testing 16 scene data set release


## Setup

### Installation
```
git clone git@github.com:ShadowIterator/4K4DGen.git
cd animating
conda env create -f environment.yml 
cd ../4dlifting
conda env create -f environment.yml 
```

### Animating
Please first set your working directory to `./animating`

#### Prepare Data
The testing panorama is in the [Google Drive](https://drive.google.com/drive/folders/18vwRuy12Nest0zqSOyobeGBuPpDTuYbO?usp=sharing).
For **Animating**, please extract the data under the `./animating/data` folder. To run on your own data, please organize the data as following:
```
data
|-- <your_own>
|   |-- <scene1>.jpg
|   |-- <scene2>.jpg
|   |-- ...
|-- <your_own>_mask
|   |-- <scene1>.png
|   |-- <scene2>.png
|   |-- ...
|-- <your_own>_config
|   |-- <scene1>.json
|   |-- <scene2>.json
|   |-- ...
```
You can also refer to the example provided in the `data` folder.

#### Prepare Checkpoint
Download the checkpoint in [googledrive](). Extract it at `animating/pretrained`.

### Lifting
Please first set your working directory to `./4dlifting`
#### Prepare Data
Put the data under the `./data` folder. Organize the files as following:
```
data
|-- <scene1>
|   |-- <scene1>.gif
|   |-- <scene1>_mask.png
|-- <scene2>
|-- ...
```
You can also refer to the `I2` folder provided as an example under the `data` folder.

#### Prepare Checkpoints
Please download the checkpoints from [googledrive](). Place them under the `./pre_checkpoints` folder



## Usage

### Animating
To run the animating phase, please run
```
cd ./animating
conda activate animating
# do animating
python gen4k.py --config ./svd_mask_config.yaml --eval validation_data.prompt_image=<path_to_image> validation_data.mask=<path_to_mask> validation_data.config=<path_to_config> pretrained_model_path=pretrained/animate_anything_svd_v1.0 --diff_mode return_latents
# decode latent codes
python gen4k.py --config ./svd_mask_config.yaml --eval validation_data.prompt_image=<path_to_image> validation_data.mask=<path_to_mask> validation_data.config=<path_to_config> pretrained_model_path=pretrained/animate_anything_svd_v1.0 --diff_mode decode_latents
```
You can also run the example bash file:
```
bash ./generate_example.sh
```
## 4D Lifting
to run 4D lifting, please refer to the following.
```
cd 4dlifting
conda activate 4dlifting
# generate initial geometry for each frame
python generate_init_geo_4k.py -s <source_path> -m <target_path>
# lifting the frames
python train.py -s <source_path>/<frame_id> -m <target_path>/<frame_id>
```

### One Command Running
using the script as following:
```
python run.py --input_img <your_image_path_or_folder>
```
If you have setup the data and environments as above, it will generate a running script `run.sh`. You can simply run `bash ./run.sh` to generate the 4D scene from your input images. For more options of the script, please see `python run.py --help`.

## Acknowledgements
We built this from [AnimateAnything](https://github.com/alibaba/animate-anything), [MultiDiffusion](https://github.com/omerbt/MultiDiffusion), and [DreamScene360](https://github.com/ShijieZhou-UCLA/DreamScene360).

## Bibtex
If you find our work useful for your project, please consider citing the following paper.
```
@misc{li20244k4dgenpanoramic4dgeneration,
      title={4K4DGen: Panoramic 4D Generation at 4K Resolution}, 
      author={Renjie Li and Panwang Pan and Bangbang Yang and Dejia Xu and Shijie Zhou and Xuanyang Zhang and Zeming Li and Achuta Kadambi and Zhangyang Wang and Zhengzhong Tu and Zhiwen Fan},
      year={2024},
      eprint={2406.13527},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.13527}, 
}
```


