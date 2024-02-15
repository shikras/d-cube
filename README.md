<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="#">
<img src=".assets/d-cube_logo.png" alt="Logo" width="310"></a>
  <h4 align="center">A detection/segmentation dataset with class names characterized by intricate and flexible expressions</h4>
    <p align="center">
    The repo is the toolbox for <b>D<sup>3</sup></b>
    <br />
    <a href="doc.md"><strong> [Doc 📚]</strong></a>
    <!-- <a href="https://huggingface.co/datasets/zbrl/d-cube"><strong> [HuggingFace 🤗]</strong></a> -->
    <a href="https://arxiv.org/abs/2307.12813"><strong> [Paper (DOD) 📄] </strong></a>
    <a href="https://arxiv.org/abs/2305.12452"><strong> [Paper (GRES) 📄] </strong></a>
    <a href="https://github.com/Charles-Xie/awesome-described-object-detection"><strong> [Awesome-DOD 🕶️] </strong></a>
    <br />
  </p>
</p>

***
Description Detection Dataset ($D^3$, /dikju:b/) is an attempt at creating a next-generation object detection dataset. Unlike traditional detection datasets, the class names of the objects are no longer simple nouns or noun phrases, but rather complex and descriptive, such as `a dog not being held by a leash`. For each image in the dataset, any object that matches the description is annotated. The dataset provides annotations such as bounding boxes and finely crafted instance masks. We believe it will contribute to computer vision and vision-language communities.



# News
- [02/14/2024] Evaluation on several SOTA methods (SPHNX (the first MLLM evaluated!), G-DINO, UNINEXT, etc.) are released, together with a [leaderboard](https://github.com/shikras/d-cube/tree/main/eval_sota) for $D^3$. :fire::fire:

- [10/12/2023] We released an [awesome-described-object-detection](https://github.com/Charles-Xie/awesome-described-object-detection) list to collect and track related works.

- [09/22/2023] Our DOD [paper](https://arxiv.org/abs/2307.12813) just got accepted by NeurIPS 2023! :fire:

- [07/25/2023] This toolkit is available on PyPI now. You can install this repo with `pip install ddd-dataset`.

- [07/25/2023] The [paper preprint](https://arxiv.org/abs/2307.12813) introducing the DOD task and the $D^3$ dataset, is available on arxiv. Check it out!

- [07/18/2023] We have released our Description Detection Dataset ($D^3$) and the first version of $D^3$ toolbox. You can download it now for your project.

- [07/14/2023] Our GRES [paper](https://arxiv.org/abs/2305.12452) has been accepted by ICCV 2023.



# Contents
- [Dataset Highlight](#task-and-dataset-highlight)
- [Download](#download)
- [Installation](#installation)
- [Usage](#usage)



# Task and Dataset Highlight

The $D^3$ dataset is meant for the Described Object Detection (DOD) task. In the image below we show the difference between Referring Expression Comprehension (REC), Object Detection/Open-Vocabulary Detection (OVD) and Described Object Detection (DOD). OVD detect object based on category name, and each category can have zero to multiple instances; REC grounds one region based on a language description, whether the object truly exits or not; DOD detect all instances on each image in the dataset, based on a flexible reference. Related works are tracked in the [awesome-DOD](https://github.com/Charles-Xie/awesome-described-object-detection) list.

![Dataset Highlight](.assets/teaser.png "Highlight of the task & dataset")

For more information on the characteristics of this dataset, please refer to our paper.



# Download
Currently we host the $D^3$ dataset on cloud drives. You can download the dataset from [Google Drive](https://drive.google.com/drive/folders/11kfY12NzKPwsliLEcIYki1yUqt7PbMEi?usp=sharing) or [Baidu Pan]().

After downloading the `d3_images.zip` (images in the dataset), `d3_pkl.zip` (dataset information for this toolkit) and `d3_json.zip` (annotation for evaluation), please extract these 3 zip files to your custom `IMG_ROOT`, `PKL_PATH` and `JSON_ANNO_PATH` directory. These paths will be used when you perform inference or evaluation on this dataset.



# Installation

## Prerequisites
This toolkit requires a few python packages like `numpy` and `pycocotools`. Other packages like `matplotlib` and `opencv-python` may also be required if you want to utilize the visualization scripts.

<!-- There are three ways to install $D^3$ toolbox, and the third one (with huggingface) is currently in the works and will be available soon. -->

There are multiple ways to install $D^3$ toolbox, as listed below:


## Install with pip
```bash
pip install ddd-dataset
```

## Install from source
```bash
git clone https://github.com/shikra/d-cube.git
# option 1: install it as a python package
cd d-cube
python -m pip install .
# done

# option 2: just put the d-cube/d_cube directory in the root directory of your local repository
```

<!-- ## Via HuggingFace Datasets 🤗
```bash
coming soon
``` -->



# Usage
Please refer to the [documentation 📚](doc.md) for more details.
Our toolbox is similar to [cocoapi](https://github.com/cocodataset/cocoapi) in style.

Here is a quick example of how to use $D^3$.
```python
from d_cube import D3
d3 = D3(IMG_ROOT, PKL_ANNO_PATH)
all_img_ids = d3.get_img_ids()  # get the image ids in the dataset
all_img_info = d3.load_imgs(all_img_ids)  # load images by passing a list of some image ids
img_path = all_img_info[0]["file_name"]  # obtain one image path so you can load it and inference
```

Some frequently asked questions are answered in [this Q&A file](./qa.md).

# Citation

If you use our $D^3$ dataset, this toolbox, or otherwise find our work valuable, please cite [our paper](https://arxiv.org/abs/2307.12813):

```bibtex
@inproceedings{xie2023DOD,
  title={Described Object Detection: Liberating Object Detection with Flexible Expressions},
  author={Xie, Chi and Zhang, Zhao and Wu, Yixuan and Zhu, Feng and Zhao, Rui and Liang, Shuang},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS)},
  year={2023}
}

@inproceedings{wu2023gres,
  title={Advancing Referring Expression Segmentation Beyond Single Image},
  author={Wu, Yixuan and Zhang, Zhao and Xie, Chi and Zhu, Feng and Zhao, Rui},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2023}
}
```

More works related to Described Object Detection are tracked in this list: [awesome-described-object-detection](https://github.com/Charles-Xie/awesome-described-object-detection).
