import numpy as np
from hog_descriptor import calculate_hog_features
from visualize_data import *
import timeit


if __name__ == '__main__':

    img = plt.imread("lena.jpg")
    hist_bins = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160])

    start = timeit.default_timer()
    hist_cells, hog_feature = calculate_hog_features(img, hist_bins)
    print(timeit.default_timer() - start)

    visualize_histogram_all_cells(hist_cells, hist_bins)
