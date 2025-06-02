# 4K4DGen: Panoramic 4D Generation at 4K Resolution

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
- [ ] Complete documents
- [x] Code release for Animation Phase
- [x] Code release for Lifting Phase
- [x] Testing 16 scene data set release


## Data
Please download the testing panorama in the [Google Drive](https://drive.google.com/drive/folders/18vwRuy12Nest0zqSOyobeGBuPpDTuYbO?usp=sharing).


## 4D Lifting
to run 4D lifting, first generate initial points.
```
python generate_init_geo_4k.py -s <source_path> -m <target_path>
```
this command will generate separate folders in \<source_path\> for each frame. 

Then lifting the frames using:
```
python train.py -s <source_path>/<frame_id> -m <target_path>/<frame_id>
```
