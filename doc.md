# $D^3$ Toolkit Documentation


## Table of Contents

- [Inference](#inference-on-d3)
- [Key Concepts](#key-concepts-for-users)
- [Evaluation Settings](#evaluation-settings)
- [Evaluation Code and Examples](#evaluation-code-and-examples)
- [Dataset statistics](#dataset-statistics)




## Inference on $D^3$

```python
# import the dataset class
from d_cube import D3
# init a dataset instance
d3 = D3(IMG_ROOT, PKL_ANNO_PATH)
all_img_ids = d3.get_img_ids()  # get the image ids in the dataset
all_img_info = d3.load_imgs(all_img_ids)  # load images by passing a list containing some image ids
img_path = all_img_info[0]["file_name"]  # obtain one image path so you can load it and inference
# then you can load the image as input for your model

group_ids = d3.get_group_ids(img_ids=[img_id])  # get the group ids by passing anno ids, image ids, etc.
sent_ids = d3.get_sent_ids(group_ids=group_ids)  # get the sentence ids by passing image ids, group ids, etc.
sent_list = d3.load_sents(sent_ids=sent_ids)
ref_list = [sent['raw_sent'] for sent in sent_list]  # list[str]
# use these language references in `ref_list` as the references to your REC/OVD/DOD model

# save the result to a JSON file
```

Concepts and structures of `anno`, `image`, `sent` and `group` are explained in [this part](#key-concepts-for-users).

In [this directory](eval_sota/) we provide the inference (and evaluation) script on some existing SOTA OVD/REC methods.



### Output Format
When the inference is done, you need to save a JSON file in the format below (COCO standard output JSON form):
```json
[
    {
        "category_id": "int, the value of sent_id, range [1, 422]",
        "bbox": "list[int], [x1, y1, w, h], predicted by your model, same as COCO result format, absolute value in the range of [w, h, w, h]",
        "image_id": "int, img_id, can be 0, 1, 2, ....",
        "score": "float, predicted by your model, no restriction on its absolute value range"
    }
]
```
This JSON file should contain a list, where each item in the list is a dictionary of one detection result.

With this JSON saved, you can evaluate the JSON in the next step. See [the evaluation step](#evaluation-code-and-examples).





## Key Concepts for Users

### `anno`
A Python dictionary where the keys are integers and the values are dictionaries with the following key-value pairs:

* `id`: an integer representing the ID of the annotation.
* `sent_id`: a list of integers representing the IDs of sentences associated with this annotation.
* `segmentation`: a Run Length Encoding (RLE) representation of the annotation.
* `area`: an integer representing the area of the annotation.
* `iscrowd`: an integer indicating whether this annotation represents a crowd or not.
* `image_id`: an integer representing the ID of the image associated with this annotation.
* `bbox`: a list of four integers representing the bounding box coordinates of the annotation in the format [x, y, width, height].
* `group_id`: a value that can be any object and represents the ID of the group associated with this annotation.

``` python
{
    1 : {
        "id": int,
        "sent_id": list,
        "segmentation": RLE,
        "area": int,
        "iscrowd": int,
        "image_id": int,
        "bbox": list, # [x, y, width, height]
        "group_id": int
    }
}
```

### `image`
A Python dictionary where the keys are integers and the values are dictionaries with the following key-value pairs:

* `id`: an integer representing the ID of the image.
* `file_name`: a string representing the file name of the image.
* `height`: an integer representing the height of the image.
* `width`: an integer representing the width of the image.
* `flickr_url`: a string representing the Flickr URL of the image.
* `anno_id`: a list of integers representing the IDs of annotations associated with this image.
* `group_id`: an integer representing the ID of the group associated with this image.
* `license`: a string representing the license of the image.

``` python
{
    int : {
        "id": int,
        "file_name": str,
        "height": int,
        "width": int,
        "flickr_url": str,
        "anno_id": list,
        "group_id": int,
        "license": str,
    }
}
```

### `sent`
A Python dictionary where the keys are integers and the values are dictionaries with the following key-value pairs:

* `id`: an integer representing the ID of the sentence.
* `anno_id`: a list of integers representing the IDs of annotations associated with this sentence.
* `group_id`: a list of integers representing the IDs of groups associated with this sentence.
* `is_negative`: a boolean indicating whether this sentence is *absence expression* or not. `True` means *absence expression*.
* `raw_sent`: a string representing the raw text of the sentence in English.
* `raw_sent_zh`: a string representing the raw text of the sentence in Chinese.

``` python
{
    int : {
        "id": int,
        "anno_id": list,  
        "group_id": list,
        "is_negative": bool,
        "raw_sent": str,
        "raw_sent_zh": str
    }
}
```

### `group`
A Python dictionary where the keys are integers and the values are dictionaries with the following key-value pairs:

* `id`: an integer representing the ID of the group.
* `pos_sent_id`: a list of integers representing the IDs of sentences that has referred obejct in the group.
* `inner_sent_id`: a list of integers representing the IDs of sentences belonging to this group.
* `outer_sent_id`: a list of integers representing the IDs of outer-group sentences that has referred obejct in the group.
* `img_id`: a list of integers representing the IDs of images of this group.
* `scene`: a list of strings representing the scenes of this group.
* `group_name`: a string representing the name of this group in English.
* `group_name_zh`: a string representing the name of this group in Chinese.

``` python
{
    int : {
        "id": int,
        "pos_sent_id": list,
        "inner_sent_id": list,
        "outer_sent_id": list,
        "img_id": list,
        "scene": list,
        "group_name": str,
        "group_name_zh": str
    }
}
```





## Evaluation Settings


### Intra- or Inter-Group Settings

The default evaluation protocol is the intra-group setting, where only a certain references are evaluated for each image.

In the $D^3$ dataset, images are collected for different groups (scenarios), and the categories (descriptions) are designed based on the scenarios. For the intra-group setting, each image are only evaluated with the descriptions from the group the image belongs to. We call this **intra-scenario setting**.

Note that each category is actually annotated on each image (with positive or negative instances).
So you can also evaluate all categories on all images, just like traditional detection datasets. We call this **inter-scenario setting**.
This is quite challenging for the DOD task as this will produce many false positive instances on current methods.

For intra-group evaluation, you should use:
```
sent_ids = d3.get_sent_ids(group_ids=group_ids)
# only get the refs (sents) for the group the image belongs to, which is usually 4
```

For inter-group evaluation, change the correponding code to:

```
sent_ids = d3.get_sent_ids()
# get all the refs in the dataset
```

This will use all the sentences in the dataset, rather than a few sentences in the group that this image belongs to.

This is the only difference in the implentation and evaluation. No further code changes need to be applied.

For more information, you can refer to the Section 3.4 of the DOD paper.


### FULL, PRES and ABS

FULL, PRES and ABS means the full descriptions (422 categories), presence descriptions (316 categories) and absence descriptions (106 categories).

The meaning of absence descriptions are the descriptions involving the absence of some concepts, like lacking certain relationships, attributes or objects. For example, descriptions like "dog *without* leash", "person *without* helmet" and "a hat that is *not* blue" are absence ones.
Similary, the descriptions involving *only* the presence of some concepts are presence descriptions.

Most existing REC datasets have presence descriptions but few absence descriptions.

For more details and the meaning of evaluating absence descriptions, please refer to Section 3.1 of the DOD paper.




## Evaluation Code and Examples

In this part, we introduce how to evaluate the performance and get the metric values given the prediction result of a JSON file.

### Write a Snippet in Your Code

This is based on [cocoapi (pycocotools)](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI), and is quite simple:

```python
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Eval results
coco = COCO(gt_path)  # `gt_path` is the ground-truth JSON path (different JSON for FULL, PRES or ABS settings in our paper)
d3_model = coco.loadRes(pred_path)  # `pred_path` is the prediction JSON file 
cocoEval = COCOeval(coco, d3_model, "bbox")
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
```

### An Off-the-shelf Script

We also provide [a script](scripts/eval_and_analysis_json.py) that can produce the evaluation results (and some additional analysis) in our paper, given a prediction JSON.
You can use it by:
```shell
python eval_and_analysis_json.py YOUR_PREDICTION_JSON_PATH
```

A few options are provided for format conversion or more analysis:
```shell
python eval_and_analysis_json.py --help

usage: An example script for $D^3$ evaluation with prediction file (JSON) [-h] [--partition-by-nbox] [--partition-by-lens] [--xyxy2xywh] pred_path

positional arguments:
  pred_path            path to prediction json

optional arguments:
  -h, --help           show this help message and exit
  --partition-by-nbox  divide the images by num of boxes for each ref
  --partition-by-lens  divide the references by their lengths
  --xyxy2xywh          transform box coords from xyxy to xywh
```


### Evaluation Examples on SOTA Methods

See [this directory](eval_sota/) for details. We include the evaluation scripts of some methods there.



## Dataset Statistics

[A python script](scripts/get_d3_stat.py) is provided for calculating the statistics of $D^3$ or visualizing figures like histograms, word clouds, etc.

The specific statistics of the dataset are available in Section 3.3 of the DOD paper.
