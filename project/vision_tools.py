import os
import math
import numpy as np
import skimage.io
from skimage.measure import label
from scipy.ndimage import gaussian_laplace
from scipy.signal import convolve2d
from scipy.ndimage.morphology import binary_dilation
from skimage.morphology import disk
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
# disable interactive mode
plt.ioff()


# Indexes
X = 0
Y = 1
ANCHOR = 0
WIDTH = 1
HEIGHT = 2
D_NUM = 0
D_RECT = 1
L_LABELS = 0
L_AREAS = 1

# Constants
MARGIN_DLR = 20
EXCLUDE_MARGIN_TOP = 100
EXCLUDE_MARGIN_BOTTOM = 100
RESULTS_MASKS_FOLDER = 'results/masks'
RESULTS_FILTERS_FOLDER = 'results/filters'
RESULTS_CARDS_FOLDER = 'results/cards'
MAX_PLAYERS = 4

# make sure that the folders exist
if not os.path.exists(RESULTS_MASKS_FOLDER):
    os.makedirs(RESULTS_MASKS_FOLDER)
if not os.path.exists(RESULTS_FILTERS_FOLDER):
    os.makedirs(RESULTS_FILTERS_FOLDER)
if not os.path.exists(RESULTS_CARDS_FOLDER):
    os.makedirs(RESULTS_CARDS_FOLDER)


def card_pipeline(folder, file, verbose=False, plot=False, save=True, show=False):
    """
    Top level function to extract individual cards and their associated data
    to be identified by the CNN
    :param folder: base folder to load the card from
    :param file: file name of image to load
    :param verbose: whether to execute in verbose or silence mode
    :param plot: whether to prepare and show the plots or not
    :param save: whether to save the plot results or not
    :param show: whether to show the plots
    :return: cards a list of segmented cards associated with each player
    dealer number the player number that is dealer for this round
    """
    # create the file path
    f_name = os.path.join(folder, file)
    # file name to be used for storing (no special characters)
    file_name = f_name.replace("\\", "_").replace("/", "_")
    if verbose:
        print(file_name)

    # load the specified image
    im = load_img(folder, file)

    # exclusion mask
    im_height, im_width = im.shape[:2]
    exclusion_mask = np.ones((im_height, im_width))
    y_excl_bot = im_height - EXCLUDE_MARGIN_BOTTOM
    y_excl_top = EXCLUDE_MARGIN_TOP
    exclusion_mask[y_excl_bot:, :] = 0
    exclusion_mask[:y_excl_top, :] = 0

    # apply the green enhancement filter
    im_green = enhance_green_channel(im)

    # filter image with high pass to detect dealer
    mask_dealer = segment_img_high_pass(im_green, threshold=40, file_name=file_name,
                                        l=(y_excl_top, y_excl_bot),
                                        plot=plot, save=save, show=show)
    # suppress reflection
    mask_dealer = mask_dealer * exclusion_mask

    # detect dealer
    (top, right, bottom, left), d_plt_rect, dealer_num = detect_dealer(mask_dealer)
    if verbose:
        print(f'Dealer is : {dealer_num}')

    # filter image with LoG to detect cards borders
    mask = segment_img_log(im_green, threshold=30, file_name=file_name,
                           l=(y_excl_top, y_excl_bot),
                           plot=plot, save=save, show=show)

    # suppress reflection
    mask = mask * exclusion_mask

    # remove D
    mask[top - MARGIN_DLR:bottom + MARGIN_DLR, left - MARGIN_DLR:right + MARGIN_DLR] = 0

    # dilate the cards borders to fill any gaps
    mask = binary_dilation(mask, structure=disk(20))
    if plot:
        plt.imshow(mask, cmap='gray')
    if show:
        plt.show()
    else:
        # plt.clf()
        plt.close()

    # create dealer structure
    dealer = (dealer_num, d_plt_rect)

    # extract cards
    cards = extract_cards(im, mask, file_name, dealer, card_seg_thresh=50,
                          verbose=verbose, plot=plot, save=save, show=show)
    return [skimage.img_as_ubyte(convert_to_gray_scale(card) < 40)for card in cards], dealer


def extract_cards(im, mask, file_name, dealer, card_seg_thresh=40, num_pix_thresh=10000,
                  verbose=True, plot=True, save=True, show=True):
    """
    Extract segmented cards as well as their associated characteristics
    :param im: base image (RGB, full scale)
    :param mask: segmentation mask to detect cards borders
    :param file_name: display name of the current image
    :param dealer: structure with dealer number and dealer rectangle in plt format to plot
    :param card_seg_thresh: threshold to use for card segmentation
    :param num_pix_thresh: minimum area in pixels that features must have to be processed
    :param verbose: whether to show messages and print infos
    :param plot: whether to construct plots
    :param save: whether to save plots
    :param show: whether to show plots
    :return: list of segmentation masks of cards
    """
    # unfold dealer structure
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

    # extract each features characteristics
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
            rect, c_x, c_y, p_id = extract_obj_prop(im_label_mask, retained_item, p_pos,
                                                    verbose=verbose)
            c_points_x.append(c_x)
            c_points_y.append(c_y)
            rects.append(rect)
            retained_items.append(retained_item)
            num_pixs.append(pix_num)
            role.append(p_id)

    # check that there is only one player with the same number
    role_copy = role
    rects_copy = rects
    for p_id in range(MAX_PLAYERS):
        index_with_role = [i for i, v in enumerate(role) if v == p_id + 1]
        if len(index_with_role) > 1:
            surf = [r[WIDTH] * r[HEIGHT] for r in [rects[i] for i in index_with_role]]
            if verbose:
                print(surf)
            index_to_keep = index_with_role[np.argmax(surf)]
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

    # plot the bounding-boxes
    if plot:
        plt.figure(figsize=(24, 12))
        for i, (idx, rect) in enumerate(zip(retained_items, rects)):
            rect_patch = Rectangle(*rect, fill=False, lw=2, ec='r')
            plt.gca().add_patch(rect_patch)
            anchor = list(rect[ANCHOR])
            anchor[Y] -= 50  # offset anchor
            plt.annotate(f'Player {role[i]}', anchor, c='r')
        # add Dealer bbox
        rect_patch = Rectangle(*dealer_rect, fill=False, lw=2, ec='r')
        plt.gca().add_patch(rect_patch)
        anchor = list(dealer_rect[ANCHOR])
        anchor[Y] -= 50  # offset anchor
        plt.annotate('Dealer', anchor, c='r')
        plt.imshow(im, interpolation='none')
        plt.title(file_name)
        if save:
            plt.savefig(f'results/{file_name}', bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        else:
            # plt.clf()
            plt.close()

    cards = []
    for i in range(MAX_PLAYERS):
        idx_player = role.index(i + 1)
        if idx_player is None:
            cards.append([])
            if verbose:
                print(f'Player {i + 1} was not detected')
            continue

        anchor, r_width, r_height = rects[idx_player]
        top = anchor[Y]
        bottom = anchor[Y] + r_height
        left = anchor[X]
        right = anchor[X] + r_width
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
        if plot:
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
            if save:
                plt.savefig(f'results/cards/{file_name.split(".")[0]}_p{i + 1}.jpg',
                            bbox_inches='tight', dpi=300)
            if show:
                plt.show()
            else:
                # plt.clf()
                plt.close()
            # plt.hist(g_card.flatten(), bins=256)
            # plt.axvline(card_seg_thresh, c='r')
            # plt.show()
        if save:
            card_mask = skimage.img_as_ubyte(g_card < card_seg_thresh)
            card_name = f'results/masks/{file_name.split(".")[0]}_p{i + 1}.jpg'
            skimage.io.imsave(card_name, card_mask)
    return cards


def detect_dealer(dealer_mask):
    """
    Detect dealer based on highest number of pixels
    :param dealer_mask: Segmented mask of the image
    :return: d_rect dealer rectangle in (top, right, bottom, left) coordinates
    plt_rect dealer rectangle in (anchor, width, height) coordinates
    dealer num player which is the dealer
    """
    # label mask to identify objects
    im_label_mask = label(dealer_mask)

    # compute each objects size in pixels
    size_items = np.unique(im_label_mask, return_counts=True)

    # remove background
    size_items = np.delete(size_items, 0, axis=1)

    # compute dealer index (biggest area of pixels)
    dealer_index = size_items[L_LABELS][np.argmax(size_items[L_AREAS])]

    # compute rectangle around dealer shape
    d_rect, plt_rect = extract_rectangle(im_label_mask, dealer_index)

    # compute dealer shape's center
    c_x = plt_rect[ANCHOR][X] + plt_rect[WIDTH] // 2
    c_y = plt_rect[ANCHOR][Y] + plt_rect[HEIGHT] // 2

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
    :return: rectangle in (top, right, bottom, left) coordinates,
    rectangle in (anchor, width, height) coordinates
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


def extract_obj_prop(im_label_mask, retained_item, player_pos=None, verbose=False):
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


def segment_img_log(im_green, threshold=30, file_name=None, l=None,
                    plot=True, save=True, show=True):
    """
    Detect image borders with Laplacian of Gaussian method
    :param im_green: image with enhanced green channel
    :param threshold: threshold value to be used for the filter
    :param file_name: file name of the image
    :param l: top and bottom exclusion lines
    :param plot: whether to plot the detected objects
    :param save: whether to save the detected objects
    :param show: whether to show the detected objects
    :return: masked image
    """
    # filter with a LoG
    im_filtered = gaussian_laplace(im_green, 0.3) * (-1)

    # normalize
    im_filtered = im_filtered / im_filtered.max()
    im_filtered *= 255

    # threshold
    mask = (im_filtered > threshold)

    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 8))
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
        # plt.tight_layout()
        if save:
            plt.savefig(f'results/filters/{file_name.split(".")[0]}_LoG.jpg',
                        bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        else:
            # plt.clf()
            plt.close(fig)
    return mask


def segment_img_high_pass(im_green, threshold, file_name=None, l=None,
                          plot=True, save=True, show=True):
    """
    Detect image features with high pass filter
    :param im_green: image with enhanced green channel
    :param threshold: threshold value to be used for the filter
    :param file_name: file name of the image
    :param l: top and bottom exclusion lines
    :param plot: whether to plot the detected objects
    :param save: whether to save the detected objects
    :param show: whether to show the detected objects
    :return:
    """
    # define filter
    hp_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    # filter image with high pass filter keeping the dimensions ('same')
    im_filtered = convolve2d(im_green, hp_filter, mode='same')

    # threshold
    mask = (im_filtered > threshold)

    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 8))
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
        # plt.tight_layout()
        if save:
            plt.savefig(f'results/filters/{file_name.split(".")[0]}_High_Pass.jpg',
                        bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        else:
            # plt.clf()
            plt.close(fig)
    return mask


def load_img(folder, image):
    """
    Load image from 'folder' with name 'image'
    :param folder: folder from which to load
    :param image: image file name
    :return: loaded image
    """
    f_name = os.path.join(folder, image)
    im_uint8 = skimage.io.imread(f_name)
    im = im_uint8.astype('int')
    return im


def convert_to_gray_scale(im):
    """
    Special Gray filter to make red be like black but eliminating the white component
    :param im: input image (RGB)
    :return: segmentation ready image
    """
    g_img = np.abs(-0.2 * im[:, :, 0] + 0.7 * im[:, :, 1] + 0.1 * im[:, :, 2])
    return g_img


def enhance_green_channel(im):
    """
    Special enhancement of green channel
    :param im: input image (RGB)
    :return: green image (1 channel)
    """
    im_green = 2 * im[:, :, 1] - im[:, :, 0] - im[:, :, 2]
    return im_green * (im_green > im_green.mean())


def dist_eucl(a, b):
    """
    Compute euclidean distance
    :param a: point a
    :param b: point b
    :return: euclidean distance between a and b
    """
    return math.sqrt((a[X] - b[X]) ** 2 + (a[Y] - b[Y]) ** 2)


def compute_pts(rank_colour, dealer):
    """
    Computes the number of points for a game of 13 rounds
    :param rank_colour:
    :param dealer: list of lists of predictions, one list per round, four predictions of the type "QS", '8D', etc..
    :return: Tuple of lists, points under standard rules and under advanced rules per player
    """
    pts_standard = [0, 0, 0, 0]
    pts_advanced = [0, 0, 0, 0]

    for i in range(np.shape(rank_colour)[0]):
        curr_round_std = [0, 0, 0, 0]
        curr_round_adv = [0, 0, 0, 0]
        dealer_suit = rank_colour[i, dealer[i] - 1][1]
        for j in range(np.shape(rank_colour)[1]):
            if rank_colour[i, j][0] == 'J':
                curr_round_std[j] = 10
                curr_round_adv[j] = 10
            elif rank_colour[i, j][0] == 'Q':
                curr_round_std[j] = 11
                curr_round_adv[j] = 11
            elif rank_colour[i, j][0] == 'K':
                curr_round_std[j] = 12
                curr_round_adv[j] = 12
            else:
                curr_round_std[j] = int(rank_colour[i, j][0])
                curr_round_adv[j] = int(rank_colour[i, j][0])
            if rank_colour[i, j][1] != dealer_suit:
                curr_round_adv[j] = -1
        for j in np.flatnonzero(curr_round_std == np.max(curr_round_std)).tolist():
            pts_standard[j] += 1
        pts_advanced[np.argmax(curr_round_adv)] += 1
    return pts_standard, pts_advanced