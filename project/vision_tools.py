import os

import skimage.io
from skimage.measure import label
from scipy.ndimage import gaussian_laplace
from scipy.signal import convolve2d
from scipy.ndimage.morphology import binary_dilation
from skimage.morphology import disk
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import math
import numpy as np

X = 0
Y = 1
MARGIN_DEALER = 50
EXCLUDE_MARGIN_TOP = 100
EXCLUDE_MARGIN_BOTTOM = 100


def dist_eucl(a, b):
    """
    Euclidean distance
    :param a: point a
    :param b: point b
    :return: euclidean distance between a and b
    """
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def detect_dealer(dealer_mask, num_pix_thresh=10000):
    im_label_mask, num_items = label(dealer_mask, return_num=True)

    size_items = np.unique(im_label_mask, return_counts=True)

    # remove background
    size_items = np.delete(size_items, 0, axis=1)

    dealer_index = size_items[0][np.argmax(size_items[1])]

    rect = extract_rectangle(im_label_mask, dealer_index)
    d_rect, plt_rect = rect
    c_x = d_rect[1] - d_rect[3]
    c_y = d_rect[2] - d_rect[0]

    # define player positons on the center of each border
    im_height, im_width = dealer_mask.shape

    p_pos = [(im_width // 2, im_height),  # 1
             (im_width, im_height // 2),  # 2
             (im_width // 2, 0),  # 3
             (0, im_height // 2)]  # 4

    dealer_num = np.argmin([dist_eucl((c_x, c_y), p) for p in p_pos]) + 1
    return d_rect, plt_rect, dealer_num


def extract_rectangle(im_label_mask, retained_item):
    """
    Extract interesting object properties
    :param im_label_mask: image labels matrix
    :param retained_item: value of label for the object
    :param player_pos: list of player positions
    :param verbose: verbose mode
    :return:
    """
    # coordinates of the points belonging to this item
    coords_y, coords_x = np.where(im_label_mask == retained_item)
    top = coords_y.min()
    right = coords_x.max()
    bot = coords_y.max()
    left = coords_x.min()

    # xy anchor for plt Rectangle patch
    anchor = (left, top)
    width = right - left
    height = bot - top
    rect_plt = (anchor, width, height)
    return (top, right, bot, left), rect_plt


def extract_obj_caracteristics(im_label_mask, retained_item, player_pos=None, verbose=False):
    """
    Extract interesting object properties
    :param im_label_mask: image labels matrix
    :param index: value of label for the object
    :param player_pos: list of player positions
    :param verbose: verbose mode
    :return:
    """
    # coordinates of the points belonging to this item
    coords_y, coords_x = np.where(im_label_mask == retained_item)
    top = coords_y.min()
    right = coords_x.max()
    bot = coords_y.max()
    left = coords_x.min()

    # xy anchor for plt Rectangle patch
    anchor = (left, top)
    width = right - left
    height = bot - top
    rect = (anchor, width, height)

    #  Center point of the item
    center_x, center_y = coords_x.mean(), coords_y.mean()

    # decide which player it is
    player_num = np.argmin([dist_eucl((center_x, center_y), p) for p in player_pos]) + 1
    if verbose:
        print(f'center of the card : {(center_x, center_y)}')
        print(f'anchor : {rect[0]}\nwidth : {rect[1]}\nheight : {rect[2]}')
        print(f'player : {player_num}')
        print('-' * 30)
    return rect, coords_x.mean(), coords_y.mean(), player_num


def find_small_d(c_x, c_y, d_index, big_value):
    # find other part of D
    dealer_x = c_x[d_index]
    dealer_y = c_y[d_index]
    dist_to_big_d = []
    for i, (a_x, a_y) in enumerate(zip(c_x, c_y)):
        if i != d_index:
            dist_to_big_d.append(dist_eucl((a_x, a_y), (dealer_x, dealer_y)))
        else:
            dist_to_big_d.append(big_value)
    return np.argmin(dist_to_big_d)


def card_pipeline(folder, file, verbose=False):
    f_name = os.path.join(folder, file)
    file_name = f_name.replace("\\", "_").replace("/", "_")
    if verbose:
        print(file_name)
    im = load_img(folder, file)

    # dilation mask
    im_height, im_width = im.shape[:2]
    dilation_mask = np.ones((im_height, im_width))
    y_excl_bot = im_height - EXCLUDE_MARGIN_BOTTOM
    y_excl_top = EXCLUDE_MARGIN_TOP
    dilation_mask[y_excl_bot:, :] = 0
    dilation_mask[:y_excl_top, :] = 0

    im_green = filter_on_green(im)

    mask_dealer = detect_object_border_dealer(im_green, threshold=40, file_name=file_name,
                                              l=(y_excl_top, y_excl_bot))
    # suppress reflection
    mask_dealer = mask_dealer * dilation_mask

    # detect dealer
    (top, right, bottom, left), d_plt_rect, dealer_num = detect_dealer(mask_dealer)
    if verbose:
        print(dealer_num)
    mask = detect_object_border(im_green, threshold=30, l=(y_excl_top, y_excl_bot))
    # suppress reflection
    mask = mask * dilation_mask

    # remove D
    mask[top - MARGIN_DEALER:bottom + MARGIN_DEALER, left - MARGIN_DEALER:right + MARGIN_DEALER] = 0
    mask = binary_dilation(mask, structure=disk(20))
    plt.imshow(mask, cmap='gray')
    plt.show()

    # extract cards
    dealer = (dealer_num, d_plt_rect)
    cards = extract_cards(im, mask, file_name, dealer, card_seg_thresh=50, verbose=True, plot_cards=True)


def extract_cards(im, mask, file_name, dealer, card_seg_thresh=40, num_pix_thresh=10000,
                  verbose=True, plot=True, plot_cards=False):
    dealer_num, dealer_rect = dealer
    # label items in image
    im_label_mask, num_items = label(mask, return_num=True)

    im_height, im_width = mask.shape
    if verbose:
        print(f'image dimensions : (width = {im_width}, height = {im_height})\n')

    # define player positons on the center of each border
    p_pos = [(im_width // 2, im_height),  # 1
             (im_width, im_height // 2),  # 2
             (im_width // 2, 0),  # 3
             (0, im_height // 2)]  # 4

    c_points_x = []
    c_points_y = []
    rects = []
    retained_items = []
    num_pixs = []
    role = []
    for i in range(num_items):
        pix_num = (im_label_mask == i + 1).sum()
        if pix_num > num_pix_thresh:
            # items that are big enough
            retained_item = i + 1
            rect, c_x, c_y, p_id = extract_obj_caracteristics(im_label_mask, retained_item, p_pos, verbose=verbose)
            c_points_x.append(c_x)
            c_points_y.append(c_y)
            rects.append(rect)
            retained_items.append(retained_item)
            num_pixs.append(pix_num)
            role.append(p_id)

    # check that there is only one player with the same number
    role_copy = role
    rects_copy = rects
    for p_id in range(4):
        index_with_role = [i for i, v in enumerate(role) if v == p_id+1]
        if len(index_with_role) > 1:
            surf = [v[1]*v[2] for v in [rects[i] for i in index_with_role]]
            print(surf)
            max_surf_index = np.argmax(surf)
            index_to_keep = index_with_role[max_surf_index]
            for i in index_with_role[::-1]:
                if i != index_to_keep:
                    del retained_items[i]
                    del rects_copy[i]
                    del c_points_x[i]
                    del c_points_y[i]
                    del num_pixs[i]
                    del role_copy[i]
    role = role_copy
    rects = rects_copy

    # plot the b-boxes
    if plot:
        plt.figure(figsize=(24, 12))
        # plt.subplot(121)
        for i, (idx, rect) in enumerate(zip(retained_items, rects)):
            rect_patch = Rectangle(*rect, fill=False, lw=2, ec='r')
            plt.gca().add_patch(rect_patch)
            anchor = list(rect[0])
            anchor[1] -= 50  # offset anchor
            plt.annotate(f'Player {role[i]}', anchor, c='r')
        rect_patch = Rectangle(*dealer_rect, fill=False, lw=2, ec='r')
        plt.gca().add_patch(rect_patch)
        anchor = list(dealer_rect[0])
        anchor[1] -= 50  # offset anchor
        plt.annotate('Dealer', anchor, c='r')
        plt.imshow(im, interpolation='none')
        plt.title(file_name)
        # plt.subplot(122)
        # plt.imshow(mask, cmap='gray', interpolation='none')
        plt.savefig(f'results/{file_name}', bbox_inches='tight', dpi=300)
        plt.show()

    cards = []
    for i in range(4):
        idx_player = role.index(i + 1)
        if idx_player is None:
            cards.append([])
            if verbose:
                print(f'Player {i+1} was not detected')
            continue
        # label_player = retained_items[idx_player]
        anchor, r_width, r_height = rects[idx_player]
        if verbose:
            print(anchor, r_width, r_height)
        top = anchor[1]
        bottom = anchor[1] + r_height
        left = anchor[0]
        right = anchor[0] + r_width
        card = im[top:bottom, left: right, :]

        # apply rotations
        if i == 1:  # rotate clock wise 90°
            card = np.transpose(card[::-1, ...], (1, 0, 2))
            r_width, r_height = r_height, r_width
        elif i == 2:  # rotate 180°
            card = card[::-1, ::-1, :]
        elif i == 3:  # rotate -90°
            card = np.transpose(card, (1, 0, 2))[::-1, ...]
            r_width, r_height = r_height, r_width

        cards.append(card)
        g_card = convert_to_gray_scale(card)
        if plot_cards:
            plt.subplot(131)
            plt.imshow(card, vmin=0, vmax=255, interpolation="none")
            plt.subplot(132)
            plt.imshow(g_card, cmap='gray', vmin=0, vmax=255, interpolation="none")
            plt.subplot(133)
            plt.imshow(g_card < card_seg_thresh, cmap='gray', interpolation="none")
            plt.axhline(0.25 * r_height, c='r')
            plt.axhline(0.75 * r_height, c='r')
            plt.axvline(0.2 * r_width, c='r')
            plt.axvline(0.8 * r_width, c='r')
            plt.tight_layout()
            plt.savefig(f'results/{file_name.split(".")[0]}_p{i+1}.jpg', bbox_inches='tight', dpi=300)
            plt.show()
            # plt.hist(g_card.flatten(), bins=256)
            # plt.axvline(card_seg_thresh, c='r')
            # plt.show()
    return cards


def convert_to_gray_scale(im):
    g_img = np.abs(-0.2 * im[:, :, 0] + 0.7 * im[:, :, 1] + 0.1 * im[:, :, 2])
    return g_img


def load_img(folder, image):
    f_name = os.path.join(folder, image)
    im_uint8 = skimage.io.imread(f_name)
    im = im_uint8.astype('int')
    return im


def filter_on_green(im):
    im_green = 2 * im[:, :, 1] - im[:, :, 0] - im[:, :, 2]
    return im_green * (im_green > im_green.mean())


def detect_object_border(im_green, threshold=30, file_name=None, l=None):
    """
    Detect image borders with difference of Gaussians method
    :param im_green: image with enhanced green channel
    :param threshold: threshold value to be used for the filter (default: 20)
    :return: masked image
    """
    im_filtered = gaussian_laplace(im_green, 0.3)*(-1)
    im_filtered = im_filtered / im_filtered.max()
    im_filtered *= 255
    # threshold
    mask = (im_filtered > threshold)

    if file_name:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 8))
        # fig.suptitle(file_name)
        ax1.imshow(im_green, cmap='gray')
        ax1.set_title('image after filtering on green')
        ax1.axis('off')
        ax2.imshow(im_filtered, cmap='gray')
        ax2.set_title('image after LoG')
        ax2.axis('off')
        ax3.imshow(mask, cmap='gray')
        ax3.set_title('image after thresholding')
        ax3.axis('off')
        for y in l:
            plt.axhline(y, c='r')
        plt.tight_layout()
        plt.savefig(f'results/{file_name.split(".")[0]}_LoG.jpg', bbox_inches='tight', dpi=300)
        plt.show()
    return mask


def detect_object_border_dealer(im_green, threshold, file_name=None, l=None):
    hp_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    im_filtered = convolve2d(im_green, hp_filter, mode='same')

    # threshold
    mask = (im_filtered > threshold)

    if file_name:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 8))
        # fig.suptitle(file_name)
        ax1.imshow(im_green, cmap='gray')
        ax1.set_title('image after filtering on green')
        ax1.axis('off')
        ax2.imshow(im_filtered, cmap='gray')
        ax2.set_title('image after filtering with high pass')
        ax2.axis('off')
        ax3.imshow(mask, cmap='gray')
        ax3.set_title('image after thresholding')
        ax3.axis('off')
        for y in l:
            plt.axhline(y, c='r')
        plt.tight_layout()
        plt.savefig(f'results/{file_name.split(".")[0]}_High_Pass.jpg', bbox_inches='tight', dpi=300)
        plt.show()
    return mask
