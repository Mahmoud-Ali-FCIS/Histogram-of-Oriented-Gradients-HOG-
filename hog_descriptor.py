import numpy as np
from scipy import ndimage
from visualize_data import visualize_image


def calculate_gradient_sobel(img, type_sobel):
    """

    :param img: np.array(128*64) image which want calculate the hog feature
    :param type_sobel: int (0 OR 1) 0 means that calculate the horizontal gradient, 1 the vertical gradient
    :return:gradient(x||y) of image np.array(128*64)
    """
    if type_sobel == 0:
        return ndimage.sobel(img, axis=0, mode='constant')
    elif type_sobel == 1:
        return ndimage.sobel(img, axis=1, mode='constant')


def calculate_gradient_magnitude(horizontal_gradient, vertical_gradient):
    """

    :param horizontal_gradient:gradient(x) of image np.array(128*64)
    :param vertical_gradient:gradient(y) of image np.array(128*64)
    :return: magnitude of gradient of image (128*64)
    """
    horizontal_gradient_square = horizontal_gradient * horizontal_gradient
    vertical_gradient_square = vertical_gradient * vertical_gradient
    sum_squares = horizontal_gradient_square + vertical_gradient_square
    grad_magnitude = np.sqrt(sum_squares)
    return grad_magnitude


def calculate_gradient_direction(horizontal_gradient, vertical_gradient):
    """

    :param horizontal_gradient: gradient(x) of image np.array(128*64)
    :param vertical_gradient: gradient(y) of image np.array(128*64)
    :return:direction of gradient of image (128*64)
    """
    grad_direction = np.arctan2(vertical_gradient, horizontal_gradient)
    grad_direction_degree = np.rad2deg(grad_direction)
    grad_direction_abs = grad_direction_degree % 180
    return grad_direction_abs


def calculate_cell_histogram(cell_direction, cell_magnitude, hist_bins):
    """

    :param cell_direction:grad_direction of image np.array(8*8)
    :param cell_magnitude:grad_magnitude of image np.array(8*8)
    :param hist_bins: np.array contains region of bins
    :return: cell_hist : the histogram of cell np.array(9)
    """
    cell_hist = np.zeros(hist_bins.size)
    cell_size = cell_direction.shape[0]

    for row_idx in range(cell_size):
        for col_idx in range(cell_size):
            curr_direction = cell_direction[row_idx, col_idx]
            curr_magnitude = cell_magnitude[row_idx, col_idx]

            diff = np.abs(curr_direction - hist_bins)
            # find interested the 2 bins
            if curr_direction <= hist_bins[0]:
                first_bin_idx = 0
                second_bin_idx = hist_bins.size - 1
            elif curr_direction > hist_bins[-1]:
                first_bin_idx = hist_bins.size - 1
                second_bin_idx = 0
            else:
                first_bin_idx = np.where(diff == np.min(diff))[0][0]
                temp = hist_bins[[(first_bin_idx - 1) % hist_bins.size,
                                  (first_bin_idx + 1) % hist_bins.size]]
                temp2 = np.abs(curr_direction - temp)
                res = np.where(temp2 == np.min(temp2))[0][0]
                if res == 0 and first_bin_idx != 0:
                    second_bin_idx = first_bin_idx - 1
                else:
                    second_bin_idx = first_bin_idx + 1

            first_bin_value = hist_bins[first_bin_idx]
            second_bin_value = hist_bins[second_bin_idx]
            # compute the distribution of magnitude
            cell_hist[second_bin_idx] += (np.abs(curr_direction - first_bin_value) / (180.0 / hist_bins.size)) \
                                        * curr_magnitude
            cell_hist[first_bin_idx] += (np.abs(curr_direction - second_bin_value) / (180.0 / hist_bins.size)) \
                                         * curr_magnitude

    return cell_hist


def calculate_cells_histogram(grad_direction, grad_magnitude, hist_bins):
    """

    :param grad_direction: direction of gradient of image (128*64)
    :param grad_magnitude:magnitude of gradient of image (128*64)
    :param hist_bins:np.array contains region of bins
    :return: histogram_all_cells: , list contains 128 of histogram cells
    """
    hist_all_cells = []
    for y in range(0, 128, 8):
        for x in range(0, 64, 8):
            cell_direction = grad_direction[y:y+8, x:x+8]
            cell_magnitude = grad_magnitude[y:y+8, x:x+8]
            cell_hist = calculate_cell_histogram(cell_direction, cell_magnitude, hist_bins)
            hist_all_cells.append(cell_hist)
    return hist_all_cells


def calculate_block_normalization(hist_cells):
    block_normalization = []
    for y in range(0, 15):
        for x in range(0, 7):
            block = hist_cells[y:y+2, x:x+2]
            block = np.reshape(block, (36, 1))
            block_norm = l2norm(block)
            block_normalization.append(block_norm)

    return block_normalization


def l2norm(vector_of_block):

    l_norm = sum(np.power(vector_of_block, 2))
    normalization = vector_of_block / l_norm
    return normalization


def calculate_hog_features(img, hist_bins):
    """

    :param img: original image with size 128*64
    :param hist_bins: np.array contains region of bins
    :return: histogram_all_cells: , list contains 128 of histogram cells
             hog_feature :vector of feature HOG

    """
    # convert image to gray and normalize it
    if len(img.shape) == 3:
        img = np.uint16(img[:, :, 0]) / 255.0
    else:
        img = np.uint16(img) / 255.0
    visualize_image(img)

    # compute the horizontal gradient
    horizontal_gradient = calculate_gradient_sobel(img, 0)
    visualize_image(horizontal_gradient)

    # compute the vertical gradient
    vertical_gradient = calculate_gradient_sobel(img, 1)
    visualize_image(vertical_gradient)

    # compute the magnitude of gradient
    grad_magnitude = calculate_gradient_magnitude(horizontal_gradient, vertical_gradient)
    visualize_image(grad_magnitude)

    # compute the direction of gradient
    grad_direction = calculate_gradient_direction(horizontal_gradient, vertical_gradient)
    visualize_image(grad_direction)

    # compute the histogram to all cells
    histogram_all_cells = calculate_cells_histogram(grad_direction, grad_magnitude, hist_bins)

    # reshape the histogram of cells to for calculate_block_normalization the by easy
    histogram_all_cells = np.reshape(histogram_all_cells, (16, 8, 9))

    # compute block normalization
    normalizations_all_blocks = calculate_block_normalization(histogram_all_cells)

    # reshape for visualization
    histogram_all_cells = np.reshape(histogram_all_cells, (16 * 8, 9))

    hog_feature = np.reshape(normalizations_all_blocks, (105, 36))

    return histogram_all_cells, hog_feature
