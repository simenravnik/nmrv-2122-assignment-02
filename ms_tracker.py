import numpy as np
import cv2

from ex2_utils import get_patch, create_epanechnik_kernel, extract_histogram, backproject_histogram, Tracker
import math


def normalize(hist):
    sum_hist = sum(hist)
    return np.array([i / sum_hist for i in hist])


def mean_shift(frame, kernel, position, nbins, q, size, eps):

    # Caculate the derivatives of size of kernel
    xi_X = np.array([list(range(-math.floor(size[0] / 2), math.floor(size[0] / 2) + 1)) for i in range(size[1])])
    xi_Y = np.array([list(range(-math.floor(size[1] / 2), math.floor(size[1] / 2) + 1)) for i in range(size[0])]).T

    iter_count = 0
    positions = []

    while True:

        # Cut patch of the new position
        patch, _ = get_patch(frame, position, size)
        
        # Extract histogram p
        p = normalize(extract_histogram(patch, nbins, weights=kernel))

        # Calculate weights v
        v = np.sqrt(np.divide(q, (p + 0.001)))

        # Calculate wi using backprojection
        wi = backproject_histogram(patch, v, nbins)

        # Calculate the changes in both x and y directions
        xk_X = np.sum(xi_X * wi) / np.sum(wi)
        xk_Y = np.sum(xi_Y * wi) / np.sum(wi)

        # Update the position accordingly
        position = (position[0] + xk_X, position[1] + xk_Y)
        positions.append(tuple(map(int, np.floor(position))))

        # Check if the algorithm converged
        if math.sqrt(xk_X ** 2 + xk_X ** 2) < eps:
            break

        if iter_count >= 20:
            break

        iter_count += 1

    return position, iter_count, positions


class MeanShiftTracker(Tracker):

    def initialize(self, image, region):
        
        self.position = (region[0] + region[2] // 2, region[1] + region[3] // 2)
        self.size = (math.floor(region[2]) // 2 * 2 + 1, math.floor(region[3]) // 2 * 2 + 1)
        self.patch, _ = get_patch(image, self.position, self.size)

        # Epanechnikov kernel
        self.kernel = create_epanechnik_kernel(self.size[0], self.size[1], self.parameters.sigma)

        # Initial value of q (Normalised extracted histogram)
        self.q = normalize(extract_histogram(self.patch, self.parameters.nbins, weights=self.kernel))

    def track(self, image):

        # Calculate new positions of the patch using mean shift algorithm
        new_position, iter_count, _ = mean_shift(image, self.kernel, self.position, self.parameters.nbins, self.q, self.size, self.parameters.eps)

        # Update current patch
        self.patch, _ = get_patch(image, new_position, self.size)

        # Update q
        hist = normalize(extract_histogram(self.patch, self.parameters.nbins, weights=self.kernel))
        self.q = (1 - self.parameters.alpha) * self.q + self.parameters.alpha * hist
        
        # Update position values
        self.position = new_position

        # Return the values
        return [int(new_position[0] - self.size[0] // 2), int(new_position[1] - self.size[1] // 2), self.size[0], self.size[1]]


class MSParams():
    def __init__(self):
        self.enlarge_factor = 1
        self.nbins = 16
        self.eps = 1
        self.sigma = 0.4
        self.alpha = 0.05
