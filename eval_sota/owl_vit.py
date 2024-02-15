import json
import os
from collections import defaultdict

from tqdm import tqdm
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

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
        outputs=outputs, target_sizes=target_size, threshold=0.1
        # the post precessing threshold will affect the performance obviously
        # you may tune it to get better performance, e.g., 0.05
    )
    boxes, scores, labels = (
        results[0]["boxes"],
        results[0]["scores"],
        results[0]["labels"],
    )
    return boxes, scores, labels


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

    # comment the following if u only need intra/inter map for full/pres/abs
    # ===================== uncomment this if u need detailed analysis =====================
    # aps = cocoEval.eval["precision"][:, :, :, 0, -1]
    # category_ids = coco.getCatIds()
    # category_names = [cat["name"] for cat in coco.loadCats(category_ids)]

    # aps_lens = defaultdict(list)
    # counter_lens = defaultdict(int)
    # for i in range(len(category_names)):
    #     ap = aps[:, :, i]
    #     ap_value = ap[ap > -1].mean()
    #     if not np.isnan(ap_value):
    #         len_ref = len(category_names[i].split(" "))
    #         aps_lens[len_ref].append(ap_value)
    #         counter_lens[len_ref] += 1

    # ap_sum_short = sum([sum(aps_lens[i]) for i in range(0, 4)])
    # ap_sum_mid = sum([sum(aps_lens[i]) for i in range(4, 7)])
    # ap_sum_long = sum([sum(aps_lens[i]) for i in range(7, 10)])
    # ap_sum_very_long = sum(
    #     [sum(aps_lens[i]) for i in range(10, max(counter_lens.keys()) + 1)]
    # )
    # c_sum_short = sum([counter_lens[i] for i in range(1, 4)])
    # c_sum_mid = sum([counter_lens[i] for i in range(4, 7)])
    # c_sum_long = sum([counter_lens[i] for i in range(7, 10)])
    # c_sum_very_long = sum(
    #     [counter_lens[i] for i in range(10, max(counter_lens.keys()) + 1)]
    # )
    # map_short = ap_sum_short / c_sum_short
    # map_mid = ap_sum_mid / c_sum_mid
    # map_long = ap_sum_long / c_sum_long
    # map_very_long = ap_sum_very_long / c_sum_very_long
    # print(
    #     f"mAP over reference length: short - {map_short:.4f}, mid - {map_mid:.4f}, long - {map_long:.4f}, very long - {map_very_long:.4f}"
    # )
    # ===================== uncomment this if u need detailed analysis =====================


def inference_on_d3(data_iter, model):
    pred = []
    error = []
    for img_id, image_path in tqdm(data_iter):
        image = load_image_general(image_path)

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

        try:
            boxes, scores, labels = get_prediction(model, image, text_list, cpu_only=False)
            for box, score, label in zip(boxes, scores, labels):
                pred_item = {
                    "image_id": img_id,
                    "category_id": sent_ids[label],
                    "bbox": convert_to_xywh(box.tolist()),  # use xywh
                    "score": float(score),
                }
                pred.append(pred_item)  # the output to be saved to JSON.
        except:
            print("error!!!")
    return pred, error


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
    assert IMG_ROOT is not None, "Please set IMG_ROOT in the script first"
    assert JSON_ANNO_PATH is not None, "Please set JSON_ANNO_PATH in the script first"
    assert PKL_ANNO_PATH is not None, "Please set PKL_ANNO_PATH in the script first"

    d3 = D3(IMG_ROOT, PKL_ANNO_PATH)

    output_dir = "ovd/owlvit/"
    os.makedirs(output_dir, exist_ok=True)

    # model prediction
    processor = OwlViTProcessor.from_pretrained("owl-vit")
    model = OwlViTForObjectDetection.from_pretrained("owl-vit")
    data_iter = get_dataset_iter(d3)
    pred, error = inference_on_d3(data_iter, model)

    pred_path = os.path.join(output_dir, f"prediction.json")
    pred_path_error = os.path.join(output_dir, "error.json")
    write_json(pred_path, pred)
    write_json(pred_path_error, error)
    # see https://github.com/shikras/d-cube/blob/main/doc.md#output-format for the output format
    # the output format is identical to COCO.

    eval_on_d3(pred_path, mode="pn")  # the FULL setting
    eval_on_d3(pred_path, mode="p")  # the PRES setting
    eval_on_d3(pred_path, mode="n")  # the ABS setting
