# -*- coding: utf-8 -*-
__author__ = "Chi Xie and Zhao Zhang"
__maintainer__ = "Chi Xie"
# data utility functions are defined in the script
import os
import json
import pickle
from collections import Counter
import shutil

# from io import StringIO
# import string

import numpy as np
import cv2
from pycocotools import mask as cocomask
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from wordcloud import WordCloud

# from pycirclize import Circos
# from Bio.Phylo.BaseTree import Tree
# from Bio import Phylo
# from newick import Node

VOC_COLORMAP = [
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]


def visualize_bbox_on_image(img, bbox_list, save_path=None, thickness=3):
    img_copy = img.copy()
    for i, bbox in enumerate(bbox_list):
        color = tuple(VOC_COLORMAP[i % len(VOC_COLORMAP)])
        x, y, w, h = bbox
        img_copy = cv2.rectangle(
            img_copy, (int(x), int(y)), (int((x + w)), int(y + h)), color, thickness
        )
    if save_path:
        cv2.imwrite(save_path, img_copy)
    return img_copy


def rle2bmask(rle):
    bm = cocomask.decode(rle)
    if len(bm.shape) == 3:
        bm = np.sum(
            bm, axis=2
        )  # sometimes there are multiple binary map (corresponding to multiple segs)
    bm = bm.astype(np.uint8)  # convert to np.uint8
    return bm


def merge_rle(rle_list, is_instance=True, on_image=False):
    if is_instance:
        cm_list = []
        for rle_idx, rle in enumerate(rle_list):
            color = VOC_COLORMAP[rle_idx]
            bm = rle2bmask(rle)
            cm = cv2.cvtColor(bm, cv2.COLOR_GRAY2BGR)
            cm_list.append(cm * color)
        merge_map = np.sum(cm_list, axis=0, dtype=np.uint8)
    else:
        bm_list = [rle2bmask(rle) for rle in rle_list]
        merge_map = np.sum(bm_list, axis=0, dtype=np.uint8)
    merge_map[merge_map >= 1] = 1
    if not on_image:
        color = VOC_COLORMAP[0]
        merge_map = cv2.cvtColor(merge_map, cv2.COLOR_GRAY2BGR)
        merge_map *= np.array(color, dtype=np.uint8)

    merge_map[merge_map > 255] = 255

    if not on_image:
        tmp_sum_map = np.sum(merge_map, axis=-1)
        merge_map[tmp_sum_map == 0] = 220
    return merge_map


def merge2bin(rle_list, img_h, img_w):
    if rle_list:
        bm_list = [rle2bmask(rle) for rle in rle_list]
        merge_map = np.sum(bm_list, axis=0, dtype=np.uint8)
        merge_map[merge_map >= 1] = 255
        merge_map = np.expand_dims(merge_map, axis=-1)
        return merge_map
    else:
        return np.zeros([img_h, img_w, 1], dtype=np.uint8)


def paste_text(img, text):
    fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL
    overlay = img.copy()
    # fontFace = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 1
    thickness = 1
    backgroud_alpha = 0.8

    retval, baseLine = cv2.getTextSize(
        text, fontFace=fontFace, fontScale=fontScale, thickness=thickness
    )
    topleft = (0, 0)
    # bottomright = (topleft[0] + retval[0], topleft[1] + retval[1]+10)
    bottomright = (img.shape[1], topleft[1] + retval[1] + 10)

    cv2.rectangle(overlay, topleft, bottomright, thickness=-1, color=(250, 250, 250))
    img = cv2.addWeighted(overlay, backgroud_alpha, img, 1 - backgroud_alpha, 0)

    cv2.putText(
        img,
        text,
        (0, baseLine + 10),
        fontScale=fontScale,
        fontFace=fontFace,
        thickness=thickness,
        color=(10, 10, 10),
    )
    return img


def load_json(json_path, to_int=False):
    clean_res_dic = {}
    with open(json_path, "r", encoding="utf-8") as f_in:
        res_dic = json.load(f_in)

    for ikey, iv in res_dic.items():
        ikey = int(ikey.strip()) if to_int else ikey.strip()
        clean_res_dic[ikey] = iv

    return clean_res_dic


def path_map(src_path, obj_path):
    def inner_map(full_path):
        return full_path.replace(src_path, obj_path)


def save_pkl(src, obj_path):
    with open(obj_path, "wb") as f_out:
        pickle.dump(src, f_out)


def load_pkl(src_path):
    with open(src_path, "rb") as f_in:
        in_pkl = pickle.load(f_in)
    return in_pkl


def copy_file(src_path, obj_path):
    shutil.copy(src_path, obj_path)


def sentence_analysis():
    return 0


def plot_bars(names, nums, is_sort, save_path=None):
    sns.set(style="whitegrid")

    if is_sort:
        zipped = zip(nums, names)
        sort_zipped = sorted(zipped, key=lambda x: (x[0], x[1]))
        result = zip(*sort_zipped)
        nums, names = [list(x) for x in result]

    fontx = {"family": "Times New Roman", "size": 10}
    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(16, 4))
    # sns.set_palette("PuBuGn_d")
    sns.barplot(names, nums, palette=sns.cubehelix_palette(80, start=0.5, rot=-0.75))
    fig.autofmt_xdate(rotation=90)
    plt.tick_params(axis="x", labelsize=10)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname("Times New Roman") for label in labels]
    plt.tight_layout()
    plt.savefig(save_path)


def plot_hist(data, bins=10, is_norm=False, save_path=None, x=None):
    sns.set_theme(style="whitegrid", font_scale=2.0)
    ax = sns.histplot(data, bins=bins, common_norm=is_norm, kde=False)
    ax.set_xlabel(x)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def add_checkerboard_bg(image, mask, save_path=None):
    # Create a new image with the same size as the original image
    new_image = np.zeros_like(image)

    # Define the size of the checkerboard pattern
    checkerboard_size = 24

    # Loop over each pixel in the mask
    for x in range(mask.shape[1]):
        for y in range(mask.shape[0]):
            # If the pixel is transparent, draw a checkerboard pattern
            if mask[y, x] == 0:
                if (x // checkerboard_size) % 2 == (y // checkerboard_size) % 2:
                    new_image[y, x] = (255, 255, 255)
                else:
                    new_image[y, x] = (128, 128, 128)
            # Otherwise, copy the corresponding pixel from the original image
            else:
                new_image[y, x] = image[y, x]

    # Save the new image with the checkerboard background
    if save_path:
        cv2.imwrite(save_path, new_image)
    return new_image


def visualize_mask_on_image(
    img, mask, save_path=None, add_edge=False, dark_background=False
):
    # Convert the mask to a binary mask if it's not already
    if mask.max() > 1:
        mask = mask.astype(np.uint8) // 255

    # Convert the mask to a 3-channel mask if it's not already
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=2)
        mask = np.tile(mask, (1, 1, 3))

    # Create a color map for the mask
    cmap = np.array([255, 117, 44], dtype=np.uint8)
    mask_colors = mask * cmap

    # Add an opaque white edge to the mask if desired
    if add_edge:
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)
            mask = np.tile(mask, (1, 1, 3))

        kernel = np.ones((5, 5), dtype=np.uint8)
        mask_edge = cv2.erode(mask, kernel, iterations=1)
        mask_edge = mask - mask_edge

        # mask_edge = np.tile(mask_edge[:, :, np.newaxis], [1, 1, 3])
        mask_colors[mask_edge > 0] = 255

    # Overlay the mask on the masked image
    if dark_background:
        masked_img = cv2.addWeighted(img, 0.4, mask_colors, 0.6, 0)
    else:
        masked_img = img.copy()
        masked_img[mask > 0] = cv2.addWeighted(img, 0.4, mask_colors, 0.6, 0)[mask > 0]

    # Save the result to the specified path if provided
    if save_path is not None:
        cv2.imwrite(save_path, masked_img)

    return masked_img


# def visualize_mask_on_image(img, mask, save_path=None, add_edge=False):
#     # Convert the mask to a binary mask if it's not already
#     if mask.max() > 1:
#         mask = mask.astype(np.uint8) // 255

#     # Convert the mask to a 3-channel mask if it's not already
#     if len(mask.shape) == 2:
#         mask = np.expand_dims(mask, axis=2)
#         mask = np.tile(mask, (1, 1, 3))

#     # Create a color map for the mask
#     cmap = np.array([255, 117, 44], dtype=np.uint8)
#     mask_colors = mask * cmap

#     # Add an opaque white edge to the mask if desired
#     if add_edge:
#         if len(mask.shape) == 2:
#             mask = np.expand_dims(mask, axis=2)
#             mask = np.tile(mask, (1, 1, 3))

#         kernel = np.ones((5, 5), dtype=np.uint8)
#         mask_edge = cv2.erode(mask, kernel, iterations=1)
#         mask_edge = mask - mask_edge

#         # mask_edge = np.tile(mask_edge[:, :, np.newaxis], [1, 1, 3])
#         mask_colors[mask_edge > 0] = 255

#     # Overlay the mask on the masked image
#     masked_img = cv2.addWeighted(img, 0.5, mask_colors, 0.5, 0)

#     # Save the result to the specified path if provided
#     if save_path is not None:
#         cv2.imwrite(save_path, masked_img)

#     return masked_img


def generate_wordclouds(sentences, save_dir):
    """Generates word clouds for different parts of speech in a list of sentences.

    Args:
        sentences: A list of sentences.
        save_dir: The directory to save the word cloud images.
    """

    os.makedirs(save_dir, exist_ok=True)
    # Load the spacy model
    nlp = spacy.load("en_core_web_sm")

    # Define the parts of speech to include in the word clouds
    pos_to_include = ["NOUN", "VERB", "ADJ", "ADV"]

    # Process each sentence and collect the relevant words for each part of speech
    words_by_pos = {pos: [] for pos in pos_to_include}
    for sent in sentences:
        doc = nlp(sent)
        for token in doc:
            if token.pos_ in pos_to_include:
                words_by_pos[token.pos_].append(token.lemma_.lower())

    # Generate a word cloud for each part of speech
    for pos, words in words_by_pos.items():
        if len(words) == 0:
            continue  # skip parts of speech with no words

        # Count the frequency of each word
        word_counts = Counter(words)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=800,
            background_color="white",
            max_words=200,
            colormap="Set2",
            max_font_size=150,
        ).generate_from_frequencies(word_counts)

        # Save the word cloud image
        filename = f"{pos.lower()}_wordcloud.png"
        filepath = os.path.join(save_dir, filename)
        wordcloud.to_file(filepath)


# def vis_group_tree(data_dict, save_path):

#     # Create 3 randomized trees
#     tree_size_list = [60, 40, 50]
#     trees = [Tree.randomized(string.ascii_uppercase, branch_stdev=0.5) for size in tree_size_list]

#     # Initialize circos sector with 3 randomized tree size
#     sectors = {name: size for name, size in zip(list("ABC"), tree_size_list)}
#     circos = Circos(sectors, space=5)

#     colors = ["tomato", "skyblue", "limegreen"]
#     cmaps = ["bwr", "viridis", "Spectral"]
#     for idx, sector in enumerate(circos.sectors):
#         sector.text(sector.name, r=120, size=12)
#         # Plot randomized tree
#         tree = trees[idx]
#         tree_track = sector.add_track((30, 70))
#         tree_track.axis(fc=colors[idx], alpha=0.2)
#         tree_track.tree(tree, leaf_label_size=3, leaf_label_margin=21)
#         # Plot randomized bar
#         bar_track = sector.add_track((70, 90))
#         x = np.arange(0, int(sector.size)) + 0.5
#         height = np.random.randint(1, 10, int(sector.size))
#         bar_track.bar(x, height, facecolor=colors[idx], ec="grey", lw=0.5, hatch="//")

#     circos.savefig(save_path, dpi=600)

# def clean_newick_key(in_str):
#     bad_chars = [':', ';', ',', '(', ')']
#     for bad_char in bad_chars:
#         in_str = in_str.replace(bad_char, ' ')
#     return in_str

# def build_tree_from_dict(data):
#     root = Node()  # create the root node
#     for key, value in data.items():
#         node = Node(name=clean_newick_key(key)) # name doesn't need to be cleaned
#         if value is not None:
#             child_node = build_tree_from_dict(value)
#             node.add_descendant(child_node)
#         root.add_descendant(node)

#     return root


def replace_chars_in_dict_keys(d):
    """
    Replaces the characters ':', ';', ',', '(', and ')' in the keys of a nested dictionary with '_'.
    """
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = replace_chars_in_dict_keys(v)
        new_key = k.translate(str.maketrans(":;,()", "_____"))
        new_dict[new_key] = v
    return new_dict


def build_newick_tree(tree_dict):
    newick_tree = ""
    if isinstance(tree_dict, dict):
        for key, value in tree_dict.items():
            if isinstance(value, dict):
                subtree = build_newick_tree(value)
                if subtree:
                    newick_tree += "(" + subtree + ")" + key + ","
                else:
                    newick_tree += key + ","
            else:
                newick_tree += key + ":" + str(value) + ","
        newick_tree = newick_tree.rstrip(",") + ")"
        return newick_tree
    else:
        return None


# def vis_group_tree(data_dict, save_path):
#     data_dic = replace_chars_in_dict_keys(data_dict)
#     super_group_names = data_dict.keys()

#     # Create 3 randomized trees
#     tree_size_list = [60, 40, 50]
#     trees = [Phylo.read(StringIO(build_newick_tree(data_dict[super_group_name])), "newick") for super_group_name in super_group_names]

#     # Initialize circos sector with 3 randomized tree size
#     sectors = {name: size for name, size in zip(list("ABC"), tree_size_list)}
#     circos = Circos(sectors, space=5)

#     colors = ["tomato", "skyblue", "limegreen"]
#     cmaps = ["bwr", "viridis", "Spectral"]
#     for idx, sector in enumerate(circos.sectors):
#         sector.text(sector.name, r=120, size=12)
#         # Plot randomized tree
#         tree = trees[idx]
#         tree_track = sector.add_track((30, 70))
#         tree_track.axis(fc=colors[idx], alpha=0.2)
#         tree_track.tree(tree, leaf_label_size=3, leaf_label_margin=21)
#         # Plot randomized bar
#         bar_track = sector.add_track((70, 90))
#         x = np.arange(0, int(sector.size)) + 0.5
#         height = np.random.randint(1, 10, int(sector.size))
#         bar_track.bar(x, height, facecolor=colors[idx], ec="grey", lw=0.5, hatch="//")

#     circos.savefig(save_path, dpi=600)
