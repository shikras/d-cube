# Frequently Asked Questions

Q:
What's the difference between Intra-Group and Inter-Group setting in [the DOD paper](https://arxiv.org/abs/2307.12813), and how to set them?

A:
Please see [this explanation in the document](./doc.md#intra--or-inter-group-settings).



Q:
What's the meaning of and difference between FULL, PRES, and ABS?

A:
Please see [this explanation in the document](./doc.md#full-pres-and-abs).



Q:
How do I perform a visualization of ground truth or prediction on a image?

A:
You can use `d3.get_anno_ids` function and pass the `img_id` you choose as parameter to get the annotation ids for a image.
After this, you can obtain the annotation details (class ids, bboxes) with `d3.load_annos`.
