# -*- coding: utf-8 -*-
__author__ = "Chi Xie and Jie Li"
__maintainer__ = "Chi Xie"

import json
import os
from collections import defaultdict
import re

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from d_cube import D3


def write_json(json_path, json_data):
    with open(json_path, "w") as f_:
        json.dump(json_data, f_)


def read_json(json_path):
    with open(json_path, "r") as f_:
        json_data = json.load(f_)
    return json_data


def load_image_general(image_path):
    image_pil = Image.open(image_path)
    return image_pil


def extract_boxes(input_string):
    # if input_string.startswith("None"):
    #     return []
    # Define the pattern using regular expression
    pattern = r'\[([\d.,; ]+)\]'
    
    # Search for the pattern in the input string
    match = re.search(pattern, input_string)
    
    # If a match is found, extract and return the boxes as a list
    if match:
        boxes_str = match.group(1)
        boxes_list = [list(map(float, box.split(','))) for box in boxes_str.split(';')]
        return boxes_list
    else:
        return []


def get_prediction(mllm_res, image, captions, cpu_only=False):
    boxes, scores, labels = [], [], []
    width, height = image.size
    for idx, res_item in enumerate(mllm_res):
        boxes_list = extract_boxes(res_item["answer"])
        for bbox in boxes_list:
            bbox_rescaled = get_true_bbox(image.size, bbox)
            boxes.append(bbox_rescaled)
            scores.append(1.0)
            labels.append(idx)
    return boxes, scores, labels


def get_dataset_iter(coco):
    img_ids = coco.get_img_ids()
    for img_id in img_ids:
        img_info = coco.load_imgs(img_id)[0]
        file_name = img_info["file_name"]
        img_path = os.path.join(IMG_ROOT, file_name)
        yield img_id, file_name, img_path


def eval_on_d3(pred_path, mode="pn"):
    assert mode in ("pn", "p", "n")
    if mode == "pn":
        gt_path = os.path.join(JSON_ANNO_PATH, "d3_full_annotations.json")
    elif mode == "p":
        gt_path = os.path.join(JSON_ANNO_PATH, "d3_pres_annotations.json")
    else:
        gt_path = os.path.join(JSON_ANNO_PATH, "d3_abs_annotations.json")
    coco = COCO(gt_path)
    d3_res = coco.loadRes(pred_path)
    cocoEval = COCOeval(coco, d3_res, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def group_sphinx_res_by_img(inference_res):
    inference_res_by_img = defaultdict(list)
    for res_item in inference_res:
        img_path = "/".join(res_item["image_path"].split("/")[-2:])
        inference_res_by_img[img_path].append(res_item)
    inference_res_by_img = dict(inference_res_by_img)
    return inference_res_by_img


def get_true_bbox(img_size, bbox):
    width, height = img_size
    max_edge = max(height, width)
    bbox = [v * max_edge for v in bbox]
    diff = abs(width - height) // 2
    if height < width:
        bbox[1] -= diff
        bbox[3] -= diff
    else:
        bbox[0] -= diff
        bbox[2] -= diff
    return bbox


def inference_on_d3(data_iter, inference_res):
    pred = []
    inf_res_by_img = group_sphinx_res_by_img(inference_res)
    for idx, (img_id, img_name, img_path) in enumerate(data_iter):
        image = load_image_general(img_path)

        # ==================================== intra-group setting ==================================== 
        # each image is evaluated with the categories in its group (usually 4)
        group_ids = d3.get_group_ids(img_ids=[img_id])
        sent_ids = d3.get_sent_ids(group_ids=group_ids)
        # ==================================== intra-group setting ====================================
        # ==================================== inter-group setting ====================================
        # each image is evaluated with all categories in the dataset (422 for the first version of the dataset)
        # sent_ids = d3.get_sent_ids()
        # ==================================== inter-group setting ====================================
        sent_list = d3.load_sents(sent_ids=sent_ids)
        text_list = [sent["raw_sent"] for sent in sent_list]

        boxes, scores, labels = get_prediction(inf_res_by_img[img_name], image, text_list, cpu_only=False)
        for box, score, label in zip(boxes, scores, labels):
            pred_item = {
                "image_id": img_id,
                "category_id": sent_ids[label],
                "bbox": convert_to_xywh(box),  # use xywh
                "score": float(score),
            }
            pred.append(pred_item)  # the output to be saved to JSON.
    return pred


def convert_to_xywh(bbox_xyxy):
    """
    Convert top-left and bottom-right corner coordinates to [x, y, width, height] format.
    """
    x1, y1, x2, y2 = bbox_xyxy
    width = x2 - x1
    height = y2 - y1
    return [x1, y1, width, height]


if __name__ == "__main__":
    IMG_ROOT = None  # set here
    JSON_ANNO_PATH = None  # set here
    PKL_ANNO_PATH = None  # set here
    # ============================== SPHINX inference result file ===============
    SPHINX_INFERENCE_RES_PATH = None
    # You can download the SPHINX d3 inference result example from:
    # https://github.com/shikras/d-cube/files/14276682/sphinx_d3_result.json
    # For the inference process, please refer to SPHINX official repo (https://github.com/Alpha-VLLM/LLaMA2-Accessory)
    # the prompts we used are available in this JSON file
    # Thanks for the contribution from Jie Li (https://github.com/theFool32)
    # ============================== SPHINX inference result file ===============
    assert IMG_ROOT is not None, "Please set IMG_ROOT in the script first"
    assert JSON_ANNO_PATH is not None, "Please set JSON_ANNO_PATH in the script first"
    assert PKL_ANNO_PATH is not None, "Please set PKL_ANNO_PATH in the script first"

    d3 = D3(IMG_ROOT, PKL_ANNO_PATH)

    output_dir = "mllm/sphinx/"  # or whatever you prefer
    inference_res = read_json(SPHINX_INFERENCE_RES_PATH)

    # model prediction
    data_iter = get_dataset_iter(d3)
    pred = inference_on_d3(data_iter, inference_res)

    pred_path = os.path.join(output_dir, f"prediction.json")
    write_json(pred_path, pred)
    # see https://github.com/shikras/d-cube/blob/main/doc.md#output-format for the output format
    # the output format is identical to COCO.

    eval_on_d3(pred_path, mode="pn")  # the FULL setting
    eval_on_d3(pred_path, mode="p")  # the PRES setting
    eval_on_d3(pred_path, mode="n")  # the ABS setting
