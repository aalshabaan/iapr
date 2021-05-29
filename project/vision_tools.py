import os

import skimage.io
from skimage.measure import label
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import math
import numpy as np

X = 0
Y = 1


def dist_eucl(a, b):
    """
    Euclidean distance
    :param a: point a
    :param b: point b
    :return: euclidean distance between a and b
    """
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def extract_obj_caracteristics(im_label_mask, retained_item, player_pos, verbose):
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


def extract_cards(im, mask, name, threshold=40, num_pix_thresh=10000, verbose=True, plot=True, plot_cards=False):
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

    # dealer is the larges number of pixels
    dealer_index = np.argmax(num_pixs)
    dealer_player = role[dealer_index]
    role[dealer_index] = 'D'
    if verbose:
        print(f'Dealer is at index {dealer_index}')
        print(f'Dealer player is {dealer_player}')

    small_d_index = find_small_d(c_points_x, c_points_y, dealer_index, max(im_height, im_width))
    role[small_d_index] = 'd'

    # merge the 'd' with the 'D'
    label_small_d = retained_items[small_d_index]
    label_big_d = retained_items[dealer_index]
    if verbose:
        print(f'Inner part of D @ index {label_small_d} -> {label_big_d}')
    # merge the inner part and the outer part
    im_label_mask[im_label_mask == label_small_d] = label_big_d
    del rects[small_d_index]
    del retained_items[small_d_index]
    del role[small_d_index]
    num_items -= 1

    # print roles
    if verbose:
        print('Roles :')
        [print(a, b) for a, b in zip(retained_items, role)]

    # plot the b-boxes
    if plot:
        plt.figure(figsize=(24, 12))
        plt.subplot(121)
        for i, (idx, rect) in enumerate(zip(retained_items, rects)):
            rect_patch = Rectangle(*rect, fill=False, lw=2, ec='r')
            plt.gca().add_patch(rect_patch)
            anchor = list(rect[0])
            anchor[1] -= 50  # offset anchor
            if role[i] == 'D':
                plt.annotate('Dealer', anchor, c='r')
            else:
                plt.annotate(f'Player {role[i]}', anchor, c='r')
        plt.imshow(im, interpolation='none')
        plt.title(name)
        plt.subplot(122)
        plt.imshow(mask, cmap='gray', interpolation='none')
        plt.show()

    cards = []
    for i in range(4):
        idx_player = role.index(i + 1)
        label_player = retained_items[idx_player]
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
            plt.imshow(g_card < threshold, cmap='gray', interpolation="none")
            plt.axhline(0.25 * r_height, c='r')
            plt.axhline(0.75 * r_height, c='r')
            plt.axvline(0.2 * r_width, c='r')
            plt.axvline(0.8 * r_width, c='r')
            plt.show()
            plt.hist(g_card.flatten(), bins=256)
            plt.axvline(threshold, c='r')
            plt.show()
    return cards, dealer_player


def convert_to_gray_scale(im):
    g_img = np.abs(-0.2 * im[:, :, 0] + 0.7 * im[:, :, 1] + 0.1 * im[:, :, 2])
    return g_img


def detect_object_border(im, threshold=80):
    # apply green channel enhancement
    im_green = 2 * im[:, :, 1] - im[:, :, 0] - im[:, :, 2]
    # threshold
    mask = im_green >= threshold
    return mask


def load_img(folder, image):
    f_name = os.path.join(folder, image)
    im_uint8 = skimage.io.imread(f_name)
    im = im_uint8.astype('int')
    return im
