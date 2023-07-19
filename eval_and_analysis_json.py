# -*- coding: utf-8 -*-
__author__ = "Chi Xie and Zhao Zhang"
__maintainer__ = "Chi Xie"
# this script takes the result json in, and print evaluation and analysis result on D-cube (FULL/PRES/ABS, etc.)
import os
import json
import argparse
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.d3 import D3


def eval_on_d3(pred_path, mode="pn", nbox_partition=None, lref_partition=False):
    assert mode in ("pn", "p", "n")
    if mode == "pn":
        gt_path = os.path.join(JSON_ANNO_PATH, "d3_full_annotations.json")
    elif mode == "p":
        gt_path = os.path.join(JSON_ANNO_PATH, "d3_pres_annotations.json")
    else:
        gt_path = os.path.join(JSON_ANNO_PATH, "d3_abs_annotations.json")

    if nbox_partition:
        gt_path, pred_path = nbox_partition_json(gt_path, pred_path, nbox_partition)

    # Eval results
    coco = COCO(gt_path)
    d3_res = coco.loadRes(pred_path)
    cocoEval = COCOeval(coco, d3_res, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    aps = cocoEval.eval["precision"][:, :, :, 0, -1]
    category_ids = coco.getCatIds()
    category_names = [cat["name"] for cat in coco.loadCats(category_ids)]

    if lref_partition:
        aps_lens = defaultdict(list)
        counter_lens = defaultdict(int)
        for i in range(len(category_names)):
            ap = aps[:, :, i]
            ap_value = ap[ap > -1].mean()
            if not np.isnan(ap_value):
                len_ref = len(category_names[i].split(" "))
                aps_lens[len_ref].append(ap_value)
                counter_lens[len_ref] += 1

        ap_sum_short = sum([sum(aps_lens[i]) for i in range(0, 4)])
        ap_sum_mid = sum([sum(aps_lens[i]) for i in range(4, 7)])
        ap_sum_long = sum([sum(aps_lens[i]) for i in range(7, 10)])
        ap_sum_very_long = sum(
            [sum(aps_lens[i]) for i in range(10, max(counter_lens.keys()) + 1)]
        )
        c_sum_short = sum([counter_lens[i] for i in range(1, 4)])
        c_sum_mid = sum([counter_lens[i] for i in range(4, 7)])
        c_sum_long = sum([counter_lens[i] for i in range(7, 10)])
        c_sum_very_long = sum(
            [counter_lens[i] for i in range(10, max(counter_lens.keys()) + 1)]
        )
        map_short = ap_sum_short / c_sum_short
        map_mid = ap_sum_mid / c_sum_mid
        map_long = ap_sum_long / c_sum_long
        map_very_long = ap_sum_very_long / c_sum_very_long
        print(
            f"mAP over reference length: short - {map_short:.4f}, mid - {map_mid:.4f}, long - {map_long:.4f}, very long - {map_very_long:.4f}"
        )


def nbox_partition_json(gt_path, pred_path, nbox_partition):
    with open(gt_path, "r") as f_gt:
        gts = json.load(f_gt)
    with open(pred_path, "r") as f_pred:
        preds = json.load(f_pred)

    cat_obj_count = d3.bbox_num_analyze()
    annos = gts["annotations"]
    new_annos = []
    for ann in annos:
        img_id = ann["image_id"]
        category_id = ann["category_id"]
        if nbox_partition == "one" and cat_obj_count[category_id - 1, img_id] == 1:
            new_annos.append(ann)
        if nbox_partition == "multi" and cat_obj_count[category_id - 1, img_id] > 1:
            new_annos.append(ann)
        if nbox_partition == "two" and cat_obj_count[category_id - 1, img_id] == 2:
            new_annos.append(ann)
        if nbox_partition == "three" and cat_obj_count[category_id - 1, img_id] == 3:
            new_annos.append(ann)
        if nbox_partition == "four" and cat_obj_count[category_id - 1, img_id] == 4:
            new_annos.append(ann)
        if nbox_partition == "four_more" and cat_obj_count[category_id - 1, img_id] > 4:
            new_annos.append(ann)
    gts["annotations"] = new_annos
    new_gts = gts
    new_preds = []
    for prd in preds:
        img_id = prd["image_id"]
        category_id = prd["category_id"]
        if nbox_partition == "no" and cat_obj_count[category_id - 1, img_id] == 0:
            new_preds.append(prd)
        if nbox_partition == "one" and cat_obj_count[category_id - 1, img_id] == 1:
            new_preds.append(prd)
        if nbox_partition == "multi" and cat_obj_count[category_id - 1, img_id] > 1:
            new_preds.append(prd)
        if nbox_partition == "two" and cat_obj_count[category_id - 1, img_id] == 2:
            new_preds.append(prd)
        if nbox_partition == "three" and cat_obj_count[category_id - 1, img_id] == 3:
            new_preds.append(prd)
        if nbox_partition == "four" and cat_obj_count[category_id - 1, img_id] == 4:
            new_preds.append(prd)
        if nbox_partition == "four_more" and cat_obj_count[category_id - 1, img_id] > 4:
            new_preds.append(prd)

    new_gt_path = gt_path.replace(".json", f".{nbox_partition}-instance.json")
    new_pred_path = pred_path.replace(".json", f".{nbox_partition}-instance.json")
    with open(new_gt_path, "w") as fo_gt:
        json.dump(new_gts, fo_gt)
    with open(new_pred_path, "w") as fo_pred:
        json.dump(new_preds, fo_pred)
    return new_gt_path, new_pred_path


def convert_to_xywh(x1, y1, x2, y2):
    """
    Convert top-left and bottom-right corner coordinates to [x,y,width,height] format.
    """
    width = x2 - x1
    height = y2 - y1
    return x1, y1, width, height


def transform_json_boxes(pred_path):
    with open(pred_path, "r") as f_:
        res = json.load(f_)
    for item in res:
        item["bbox"] = convert_to_xywh(*item["bbox"])
    res_path = pred_path.replace(".json", ".xywh.json")
    with open(res_path, "w") as f_w:
        json.dump(res, f_w)
    return res_path


if __name__ == "__main__":
    IMG_ROOT = None  # set here
    JSON_ANNO_PATH = None  # set here
    PKL_ANNO_PATH = None  # set here
    assert IMG_ROOT is not None, "Please set IMG_ROOT in the script first"
    assert JSON_ANNO_PATH is not None, "Please set JSON_ANNO_PATH in the script first"
    assert PKL_ANNO_PATH is not None, "Please set PKL_ANNO_PATH in the script first"
    d3 = D3(IMG_ROOT, PKL_ANNO_PATH)

    parser = argparse.ArgumentParser(
        "An example script for D-cube evaluation with prediction file (JSON)",
        add_help=True,
    )
    parser.add_argument("pred_path", type=str, help="path to the prediction JSON file")
    parser.add_argument(
        "--partition-by-nbox",
        action="store_true",
        help="divide the images by num of boxes for each ref",
    )
    parser.add_argument(
        "--partition-by-lens",
        action="store_true",
        help="divide the references by their lengths",
    )
    parser.add_argument(
        "--xyxy2xywh",
        action="store_true",
        help="transform box coords from xyxy to xywh",
    )
    args = parser.parse_args()
    if args.xyxy2xywh:
        pred_path = transform_json_boxes(args.pred_path)
    else:
        pred_path = args.pred_path
    pred_path = args.pred_path
    if args.partition_by_nbox:
        # partiton: no-instance, one-instance, multi-instance
        for mode in ("pn", "p", "n"):
            # for ptt in ('no', 'one', 'multi'):
            for ptt in ("no", "one", "two", "three", "four", "four_more"):
                eval_on_d3(pred_path, mode=mode, nbox_partition=ptt)
    else:
        eval_on_d3(pred_path, mode="pn", lref_partition=args.partition_by_lens)
        eval_on_d3(pred_path, mode="p", lref_partition=args.partition_by_lens)
        eval_on_d3(pred_path, mode="n", lref_partition=args.partition_by_lens)
