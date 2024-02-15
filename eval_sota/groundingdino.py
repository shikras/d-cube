# -*- coding: utf-8 -*-
__author__ = "Chi Xie"
__maintainer__ = "Chi Xie"

# An example for how to run this script:
# CUDA_VISIBLE_DEVICES=0
# python groundingdino.py \
#     -c ./groundingdino/config/GroundingDINO_SwinB.cfg.py \
#     -p ./ckpt/groundingdino_swinb_cogcoor.pth \
#     -o "outputs/gdino_d3" \
#     --box_threshold 0.05 \
#     --text_threshold 0.05 \
#     --img-top1

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from d_cube import D3


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)
    return image_pil, mask


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, cpu_only=False):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    logits_list = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        logits_list.append(logit.max().item())
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases, logits_list


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


def inference_on_d3(data_iter, model, args, box_threshold, text_threshold):
    pred = []
    for idx, (img_id, image_path) in enumerate(tqdm(data_iter)):
        # load image
        image_pil, image = load_image(image_path)
        size = image_pil.size
        W, H = size

        group_ids = d3.get_group_ids(img_ids=[img_id])
        sent_ids = d3.get_sent_ids(group_ids=group_ids)
        sent_list = d3.load_sents(sent_ids=sent_ids)
        text_list = [sent['raw_sent'] for sent in sent_list]

        for sent_id, text_prompt in zip(sent_ids, text_list):
            # run model
            boxes_filt, pred_phrases, logit_list = get_grounding_output(
                model, image, text_prompt, box_threshold, text_threshold, cpu_only=args.cpu_only, with_logits=False,
            )
            if args.vis:
                pred_dict = {
                    "boxes": boxes_filt,  # [x_center, y_center, w, h]
                    "size": [size[1], size[0]],
                    "labels": [f"{phrase}({str(logit)[:4]})" for phrase, logit in zip(pred_phrases, logit_list)],
                }
                image_with_box = plot_boxes_to_image(image_pil.copy(), pred_dict)[0]
                image_with_box.save(os.path.join(output_dir, f"{img_id}_{text_prompt}.jpg"))
            if not logit_list:
                continue
            if args.img_top1:
                max_score_idx = logit_list.index(max(logit_list))
                bboxes, phrases, logits = [boxes_filt[max_score_idx]], [pred_phrases[max_score_idx]], [logit_list[max_score_idx]]
            else:
                bboxes, phrases, logits = boxes_filt, pred_phrases, logit_list
            for box, phrase, logit in zip(bboxes, phrases, logits):
                if len(phrase) > args.overlap_percent * len(text_prompt) or phrase == text_prompt:
                    x1, y1, w, h = box.tolist()
                    x0, y0 = x1 - w / 2, y1 - h / 2
                    pred_item = {
                        "image_id": img_id,
                        "category_id": sent_id,
                        "bbox": [x0 * W, y0 * H, w * W, h * H],
                        "score": float(logit),
                    }
                    pred.append(pred_item)

    return pred


if __name__ == "__main__":
    IMG_ROOT = None  # set here
    JSON_ANNO_PATH = None  # set here
    PKL_ANNO_PATH = None  # set here
    assert IMG_ROOT is not None, "Please set IMG_ROOT in the script first"
    assert JSON_ANNO_PATH is not None, "Please set JSON_ANNO_PATH in the script first"
    assert PKL_ANNO_PATH is not None, "Please set PKL_ANNO_PATH in the script first"

    d3 = D3(IMG_ROOT, PKL_ANNO_PATH)

    parser = argparse.ArgumentParser("Grounding DINO evaluation on D-cube (https://arxiv.org/abs/2307.12813)", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    # parser.add_argument("--image_path", "-i", type=str, required=True, help="path to image file")
    # parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )
    parser.add_argument("--vis", action="store_true", help="visualization on D3")

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    parser.add_argument("--img-top1", action="store_true", help="select only the box with top max score")
    # parser.add_argument("--overlap-percent", type=float, default=1.0, help="overlapping percentage between input prompt and output label")
    # this overlapping percentage denotes an additional post-processing technique we designed. if you turn this on, you may get higher performance by tuning this parameter.
    args = parser.parse_args()
    args.overlap_percent = 1  # by default, we do not use this technique.
    print(args)

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    # image_path = args.image_path
    # text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)

    data_iter = get_dataset_iter(d3)

    pred = inference_on_d3(data_iter, model, args, box_threshold=box_threshold, text_threshold=text_threshold)

    pred_path = os.path.join(output_dir, f"prediction.json")
    with open(pred_path, "w") as f_:
        json.dump(pred, f_)
    eval_on_d3(pred_path, mode='pn')
    eval_on_d3(pred_path, mode='p')
    eval_on_d3(pred_path, mode='n')
