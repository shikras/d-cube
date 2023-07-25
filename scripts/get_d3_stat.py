import numpy as np

from d_cube.vis_util import plot_hist
from d_cube import D3


def vis_num_instance(cat_obj_count):
    # Assuming `cat_obj_count` is your numpy array of shape [n_cat, n_img]

    # Calculate the total number of instances in each image
    total_instances_per_image = np.sum(cat_obj_count, axis=0)

    # # Plot the histogram
    # plt.hist(total_instances_per_image, bins=20)
    # plt.xlabel('Number of Instances')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Number of Instances on a Image')

    # # Save the figure
    # plt.savefig('vis_fig/instance_distribution.png', bbox_inches='tight')
    # plt.close()
    plot_hist(
        total_instances_per_image,
        bins=max(total_instances_per_image) - min(total_instances_per_image) + 1,
        save_path="vis_fig/instance_dist_hist.pdf",
    )


def vis_num_category(cat_obj_count):
    # Assuming `cat_obj_count` is your numpy array of shape [n_cat, n_img]

    # Calculate the number of categories in each image
    num_categories_per_image = np.sum(cat_obj_count > 0, axis=0)

    # # Plot the histogram
    # plt.hist(num_categories_per_image, bins=20)
    # plt.xlabel('Number of Categories')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Number of Categories on a Image')

    # # Save the figure
    # plt.savefig('vis_fig/category_distribution.png', bbox_inches='tight')
    # plt.close()
    plot_hist(
        num_categories_per_image,
        bins=max(num_categories_per_image) - min(num_categories_per_image) + 1,
        save_path="vis_fig/category_dist_hist.pdf",
    )


def vis_num_img_per_cat(cat_obj_count):
    num_img_per_cat = np.sum(cat_obj_count > 0, axis=1)
    plot_hist(
        num_img_per_cat,
        bins=20,
        save_path="vis_fig/nimg_pcat_hist.pdf",
        x="Num. of images",
    )


def vis_num_box_per_cat(cat_obj_count):
    num_box_per_cat = np.sum(cat_obj_count, axis=1)
    plot_hist(
        num_box_per_cat,
        bins=20,
        save_path="vis_fig/nbox_pcat_hist.pdf",
        x="Num. of instances",
    )


def vis_num_box_per_cat_per_img(cat_obj_count):
    img_obj_count = cat_obj_count.reshape(-1)
    plot_hist(
        img_obj_count[img_obj_count > 0],
        bins=max(img_obj_count) - min(img_obj_count) + 1,
        save_path="vis_fig/nbox_pcat_pimg_hist.pdf",
        x="Num. of instances on a image",
    )


if __name__ == "__main__":
    IMG_ROOT = None  # set here
    PKL_ANNO_PATH = None  # set here
    assert IMG_ROOT is not None, "Please set IMG_ROOT in the script first"
    assert PKL_ANNO_PATH is not None, "Please set PKL_ANNO_PATH in the script first"
    d3 = D3(IMG_ROOT, PKL_ANNO_PATH)

    cat_obj_count = d3.bbox_num_analyze()
    vis_num_instance(cat_obj_count)
    vis_num_category(cat_obj_count)
    vis_num_img_per_cat(cat_obj_count)
    vis_num_box_per_cat(cat_obj_count)
    vis_num_box_per_cat_per_img(cat_obj_count)

    d3.stat_description(with_rev=False)
    d3.stat_description(with_rev=True)
    d3.stat_description(with_rev=False, inter_group=True)
    d3.stat_description(with_rev=True, inter_group=True)
