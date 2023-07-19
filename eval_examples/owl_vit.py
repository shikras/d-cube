import json
import os
import time
import numpy as np
from collections import defaultdict
import logging

import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import torch

from d_cube import D3


def load_image_general(image_path):
    image_pil = Image.open(image_path)
    return image_pil, image_pil


def get_prediction(model, image, captions, cpu_only=False):
    for i in range(len(captions)):
        captions[i] = captions[i].lower()
        captions[i] = captions[i].strip()
        if not captions[i].endswith("."):
            captions[i] = captions[i] + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    with torch.no_grad():
        inputs = processor(text=[captions], images=image, return_tensors="pt").to(
            device
        )
        outputs = model(**inputs)
    target_size = torch.Tensor([image.size[::-1]]).to(device)
    results = processor.post_process_object_detection(
        outputs=outputs, target_sizes=target_size, threshold=0.05
    )
    return results


def get_dataset_iter(coco):
    img_ids = coco.get_img_ids()
    for img_id in img_ids:
        img_info = coco.load_imgs(img_id)[0]
        file_name = img_info["file_name"]
        img_path = os.path.join(IMG_ROOT, file_name)
        yield img_id, img_path


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

    aps = cocoEval.eval["precision"][:, :, :, 0, -1]
    category_ids = coco.getCatIds()
    category_names = [cat["name"] for cat in coco.loadCats(category_ids)]

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


def inference_on_d3(data_iter, model):
    pred = []
    error = []
    for idx, (img_id, image_path) in enumerate(data_iter):
        logging.critical(idx)
        logging.critical(time.asctime(time.localtime(time.time())))
        image_pil, image = load_image_general(image_path)

        # group
        group_ids = d3.get_group_ids(img_ids=[img_id])
        sent_ids = d3.get_sent_ids(group_ids=group_ids)
        sent_list = d3.load_sents(sent_ids=sent_ids)
        text_list = [sent["raw_sent"] for sent in sent_list]

        try:
            results = get_prediction(model, image, text_list, cpu_only=False)
            i = 0
            boxes, scores, labels = (
                results[i]["boxes"],
                results[i]["scores"],
                results[i]["labels"],
            )
            for box, score, label in zip(boxes, scores, labels):
                pred_item = {
                    "image_id": img_id,
                    "category_id": sent_ids[label],
                    "bbox": box.tolist(),
                    "score": float(score),
                }
                pred.append(pred_item)
        except:
            print("error!!!")
    return pred, error


def convert_to_xywh(x1, y1, x2, y2):
    """
    Convert top-left and bottom-right corner coordinates to x,y,width,height format.
    """
    width = x2 - x1
    height = y2 - y1
    return x1, y1, width, height


if __name__ == "__main__":
    IMG_ROOT = None  # set here
    JSON_ANNO_PATH = None  # set here
    PKL_ANNO_PATH = None  # set here
    assert IMG_ROOT is not None, "Please set IMG_ROOT in the script first"
    assert JSON_ANNO_PATH is not None, "Please set JSON_ANNO_PATH in the script first"
    assert PKL_ANNO_PATH is not None, "Please set PKL_ANNO_PATH in the script first"
    d3 = D3(IMG_ROOT, PKL_ANNO_PATH)

    output_dir = "ovd/owlvit/"
    os.makedirs(output_dir, exist_ok=True)

    # model predicting
    processor = OwlViTProcessor.from_pretrained("owl-vit")
    model = OwlViTForObjectDetection.from_pretrained("owl-vit")
    data_iter = get_dataset_iter(d3)
    pred, error = inference_on_d3(data_iter, model)

    pred_path = os.path.join(output_dir, f"eval_d3.json")
    pred_path_error = os.path.join(output_dir, "error.json")

    with open(pred_path, "w") as f_:
        json.dump(pred, f_)
    with open(pred_path_error, "w") as f2:
        json.dump(error, f2)

    # change to xywh format of bbox
    with open(pred_path, "r") as f_:
        res = json.load(f_)
    for item in res:
        item["bbox"] = convert_to_xywh(*item["bbox"])
    res_path = pred_path.replace(".json", ".xywh.json")
    with open(res_path, "w") as f_w:
        json.dump(res, f_w)

    eval_on_d3(res_path, mode="pn")
    eval_on_d3(res_path, mode="p")
    eval_on_d3(res_path, mode="n")
