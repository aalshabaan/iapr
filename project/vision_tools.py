from skimage.measure import label
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import math
import numpy as np


def dist_eucl(a, b):
    """
    Euclidean distance
    :param a: point a
    :param b: point b
    :return: euclidean distance between a and b
    """
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def extract_cards(im, mask, verbose=True, plot=True):
    # label items in image
    im_label_mask, num_items = label(mask, return_num=True)

    im_height, im_width = mask.shape
    if verbose:
        print(f'image dimensions : (width = {im_width}, height = {im_height})')

    # define player positons on the center of each border
    player_pos = [(im_width // 2, im_height),  # 1
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
        if pix_num > 10000:
            # items that are big enough
            retained_items.append(i + 1)
            num_pixs.append(pix_num)

            # coordinates of the points belonging to this item
            coords_item = np.where(im_label_mask == i + 1)
            top = coords_item[0].min()
            right = coords_item[1].max()
            bot = coords_item[0].max()
            left = coords_item[1].min()

            # xy anchor for plt Rectangle patch
            anchor = (left, top)
            width = right - left
            height = bot - top
            rect = (anchor, width, height)
            rects.append(rect)

            #  Center point of the item
            center_y, center_x = [a.mean() for a in coords_item]  # y, x
            c_points_x.append(center_x)
            c_points_y.append(center_y)
            center_point = (center_x, center_y)

            # decide which player it is
            player_num = np.argmin([dist_eucl((center_x, center_y), p) for p in player_pos]) + 1
            role.append(player_num)
            if verbose:
                print(f'item : {i + 1}')
                print(f'total number of pixels : {pix_num}')
                print(f'center of the card : {center_point}')
                print(f'anchor : {rect[0]}\nwidth : {rect[1]}\nheight : {rect[2]}')
                print(f'player : {player_num}')
                print('-' * 30)

    # dealer is the larges number of pixels
    dealer_index = np.argmax(num_pixs)
    dealer_player = role[dealer_index]
    role[dealer_index] = 'D'
    if verbose:
        print(f'Dealer is at index {dealer_index}')
        print(f'Dealer player is {dealer_player}')

    # find other part of D
    dealer_x = c_points_x[dealer_index]
    dealer_y = c_points_y[dealer_index]
    dist_to_big_d = []
    for i, (a_x, a_y) in enumerate(zip(c_points_x, c_points_y)):
        if i != dealer_index:
            dist_to_big_d.append(dist_eucl((a_x, a_y), (dealer_x, dealer_y)))
        else:
            dist_to_big_d.append(height)
    small_d_index = np.argmin(dist_to_big_d)
    role[small_d_index] = 'd'

    # merge the 'd' with the 'D'
    item_small_d = retained_items[small_d_index]
    item_big_D = retained_items[dealer_index]
    if verbose:
        print(f'Inner part of D @ index {item_small_d} -> {item_big_D}')
    # merge the inner part and the outer part
    im_label_mask[im_label_mask == item_small_d] = item_big_D
    del rects[small_d_index]
    del retained_items[small_d_index]
    del role[small_d_index]
    num_items -= 1

    if verbose:
        print('Roles :')
        [print(a, b) for a, b in zip(retained_items, role)]

    if plot:
        plt.figure(figsize=(12, 12))
        for i, (idx, rect) in enumerate(zip(retained_items, rects)):
            rect_patch = Rectangle(*rect, fill=False, lw=2, ec='r')
            plt.gca().add_patch(rect_patch)
            anchor = list(rect[0])
            anchor[1] -= 50  # offset anchor
            if role[i] == 'D':
                plt.annotate('Dealer', anchor, c='r')
            else:
                plt.annotate(f'Player {role[i]}', anchor, c='r')
        plt.imshow(im, cmap='gray')
        plt.show()

    cards = []
    for i in range(4):
        idx_player = role.index(i + 1)
        label_player = retained_items[idx_player]
        anchor, r_width, r_height = rects[idx_player]
        print(anchor, r_width, r_height)
        top = anchor[1]
        bottom = anchor[1] + r_height
        left = anchor[0]
        right = anchor[0] + r_width
        card = im[top:bottom, left: right, :]

        # apply rotations
        if i == 1:  # rotate clock wise 90°
            card = np.transpose(card[::-1, ...], (1, 0, 2))
        elif i == 2:  # rotate 180°
            card = card[::-1, ::-1, :]
        elif i == 3:  # rotate -90°
            card = np.transpose(card, (1, 0, 2))[::-1, ...]
        g_card = convert_to_gray_scale(card)
        # plt.imshow(g_card, cmap='gray')
        # plt.show()

        plt.subplot(131)
        plt.imshow(card)
        plt.subplot(132)
        plt.imshow(g_card, cmap='gray')
        plt.subplot(133)
        plt.imshow(g_card < 100, cmap='gray')
        plt.show()


def convert_to_gray_scale(im):
    g_img = 0.299 * im[:, :, 0] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 2]
    print(g_img.shape)
    return g_img
