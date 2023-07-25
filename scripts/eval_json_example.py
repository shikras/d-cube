# -*- coding: utf-8 -*-
__author__ = "Chi Xie and Zhao Zhang"
__maintainer__ = "Chi Xie"
# this script takes the result json in, and print evaluation and analysis result on D-cube (FULL/PRES/ABS, etc.)
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Eval results with COCOAPI
gt_path = "./d3_full_annotations.json"  # FULL, PRES or ABS
pred_path = None  # set your prediction JSON path
coco = COCO(gt_path)
d3_res = coco.loadRes(pred_path)
cocoEval = COCOeval(coco, d3_res, "bbox")
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
