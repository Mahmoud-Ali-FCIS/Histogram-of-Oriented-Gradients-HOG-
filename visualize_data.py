import matplotlib.pyplot as plt


def visualize_histogram_all_cells(hist_all_cells, hist_bin):
    """

    :param hist_all_cells: list contains 128 of histogram cells
    :param hist_bin: np.array contains region of bins
    :return: None
    """
    for i in range(128):
        plt.bar(hist_bin, hist_all_cells[i], align="center", width=0.8)
        plt.show()


def visualize_image(image):

    plt.imshow(image)
    plt.show()
