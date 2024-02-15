# Evaluting SOTA Methods on $D^3$

## Leaderboard

In this directory, we keep the scripts or github links (official or custom) to evaluate SOTA methods (REC/OVD/DOD/MLLM) on $D^3$:

| Name | Paper | Original Tasks | Training Data | Evaluation Code | Intra-FULL/PRES/ABS/Inter-FULL/PRES/ABS | Source | Note |
|:-----|:-----:|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| OFA-large | [OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework (ICML 2022)](https://arxiv.org/abs/2202.03052) | REC | - | - | 4.2/4.1/4.6/0.1/0.1/0.1 | [DOD paper](https://arxiv.org/abs/2307.12813) | - |
| CORA-R50 | [CORA: Adapting CLIP for Open-Vocabulary Detection with Region Prompting and Anchor Pre-Matching (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_CORA_Adapting_CLIP_for_Open-Vocabulary_Detection_With_Region_Prompting_and_CVPR_2023_paper.pdf) | OVD | - | - | 6.2/6.7/5.0/2.0/2.2/1.3 | [DOD paper](https://arxiv.org/abs/2307.12813) | - |
| OWL-ViT-large | [Simple Open-Vocabulary Object Detection with Vision Transformers (ECCV 2022)](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700714.pdf) | OVD | - | [DOD official](./owl_vit.py) | 9.6/10.7/6.4/2.5/2.9/2.1 | [DOD paper](https://arxiv.org/abs/2307.12813) | Post-processing hyper-parameters may affect the performance and the result may not exactly match the paper |
| SPHINX-7B | [SPHINX: The Joint Mixing of Weights, Tasks, and Visual Embeddings for Multi-modal Large Language Models (arxiv 2023)](https://arxiv.org/abs/2311.07575) | **MLLM** capable of REC | - | [DOD official](./sphinx.py) | 10.6/11.4/7.9/-/-/- | DOD authors | A lot of contribution from [Jie Li](https://github.com/theFool32) |
| GLIP-T | [Grounded Language-Image Pre-training (CVPR 2022)](https://arxiv.org/abs/2112.03857)  | OVD & PG | - | - | 19.1/18.3/21.5/-/-/- | GEN paper | - |
| UNINEXT-huge | [Universal Instance Perception as Object Discovery and Retrieval (CVPR 2023)](https://arxiv.org/abs/2303.06674v2) | OVD & REC | - | [DOD official](https://github.com/Charles-Xie/UNINEXT_D3) | 20.0/20.6/18.1/3.3/3.9/1.6 | [DOD paper](https://arxiv.org/abs/2307.12813) | - |
| Grounding-DINO-base | [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection (arxiv 2023)](https://arxiv.org/abs/2303.05499) | OVD & REC | - | [DOD official](./groundingdino.py) | 20.7/20.1/22.5/2.7/2.4/3.5 | [DOD paper](https://arxiv.org/abs/2307.12813) | Post-processing hyper-parameters may affect the performance and the result may not exactly match the paper |
| OFA-DOD-base | [Described Object Detection: Liberating Object Detection with Flexible Expressions (NeurIPS 2023)](https://arxiv.org/abs/2307.12813) | DOD | - | - | 21.6/23.7/15.4/5.7/6.9/2.3 | [DOD paper](https://arxiv.org/abs/2307.12813) | - |
| FIBER-B | [Coarse-to-Fine Vision-Language Pre-training with Fusion in the Backbone (NeurIPS 2022)](https://arxiv.org/abs/2206.07643) | OVD & REC | - | - | 22.7/21.5/26.0/-/-/- | GEN paper | - |
| MM-Grounding-DINO | [An Open and Comprehensive Pipeline for Unified Object Grounding and Detection (arxiv 2024)](https://arxiv.org/abs/2401.02361) | DOD & OVD & REC | O365, GoldG, GRIT, V3Det | [MM-GDINO official](https://github.com/open-mmlab/mmdetection/tree/main/configs/mm_grounding_dino#zero-shot-description-detection-datasetdod) | 22.9/21.9/26.0/-/-/- | MM-GDINO paper | - |
| GEN (FIBER-B) | [Generating Enhanced Negatives for Training Language-Based Object Detectors (arxiv 2024](https://arxiv.org/abs/2401.00094) | DOD | - | - | 26.0/25.2/28.1/-/-/- | GEN paper | Enhancement based on FIBER-B |
| APE-large (D) | [Aligning and Prompting Everything All at Once for Universal Visual Perception (arxiv 2023)](https://arxiv.org/abs/2312.02153) | DOD & OVD & REC | COCO, LVIS, O365, OpenImages, Visual Genome, RefCOCO/+/g, SA-1B, GQA, PhraseCut, Flickr30k | [APE official](https://github.com/shenyunhang/APE) | 37.5/38.8/33.9/21.0/22.0/17.9 | APE paper | Extra training data helps for this amazing performance |


Some extra notes:
- Each method is currently recorded by *the variant with the highest performance* in this table, if there are multiple variants available, so it's only a leaderboard, not meant for fair comparison.
- Methods like GLIP, FIBER, etc. are actually not evaluated on OVD benchmarks. For zero-shot eval on DOD, We currently do not distinguish between methods for OVD benchmarks and methods for ZS-OD, as long as it is verified with open-set detection capability.

For other variants (e.g. for a fair comparison regarding data, backbone, etc.), please refer to the papers.
