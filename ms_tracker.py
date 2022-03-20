from math import floor

import numpy as np
import cv2

from ex2_utils import get_patch, create_epanechnik_kernel, extract_histogram, backproject_histogram, Tracker
import math

def reshape_patch(patch):
    if patch.shape[0] % 2 == 0:
        patch = patch[:-1, :]
    if patch.shape[1] % 2 == 0:
        patch = patch[:, :-1]
    return patch


def get_odd_size(size):
    return math.ceil(size[0] / 2.) * 2 - 1, math.ceil(size[1] / 2.) * 2 - 1


def mean_shift(frame, kernel, position, nbins, q, size, eps):

    # Round to odd number
    kernel_size_x, kernel_size_y = get_odd_size(size)

    # Caculate the derivatives of size of kernel
    xi_X = np.array([list(range(-math.floor(kernel_size_x / 2), math.floor(kernel_size_x / 2) + 1)) for i in range(kernel_size_y)])
    xi_Y = np.array([list(range(-math.floor(kernel_size_y / 2), math.floor(kernel_size_y / 2) + 1)) for i in range(kernel_size_x)]).T

    iter_count = 0
    positions = []

    while True:

        # Cut patch of the new position
        patch, _ = get_patch(frame, position, (kernel_size_x, kernel_size_y))
        patch = reshape_patch(patch)
        
        # Extract histogram p
        hist = extract_histogram(patch, nbins, weights=kernel)
        hist_sum = sum(hist)
        p = np.array([i / hist_sum for i in hist])

        # Calculate weights v
        v = np.sqrt(np.divide(q, (p + eps)))

        # Calculate wi using backprojection
        wi = backproject_histogram(patch, v, nbins)

        # Calculate the changes in both x and y directions
        xk_X = np.sum(xi_X * wi) / np.sum(wi)
        xk_Y = np.sum(xi_Y * wi) / np.sum(wi)

        # Update the position accordingly
        position = (position[0] + xk_X, position[1] + xk_Y)
        positions.append(tuple(map(int, np.floor(position))))

        # Check if the algorithm converged
        if abs(xk_X) < eps and abs(xk_Y) < eps:
            break

        # Sanity check if the algorihtm does not converge
        if iter_count >= 500:
            break

        iter_count += 1

    return position, iter_count, positions


class MeanShiftTracker(Tracker):

    def initialize(self, image, region):

        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        self.window = max(region[2], region[3]) * self.parameters.enlarge_factor

        left = max(region[0], 0)
        top = max(region[1], 0)

        right = min(region[0] + region[2], image.shape[1] - 1)
        bottom = min(region[1] + region[3], image.shape[0] - 1)

        self.patch = image[int(top):int(bottom), int(left):int(right)]
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = get_odd_size((region[2], region[3]))

        # Epanechnikov kernel
        self.kernel = create_epanechnik_kernel(self.size[0], self.size[1], self.parameters.sigma)

        # Initial value of q (Normalised extracted histogram)
        hist = extract_histogram(reshape_patch(self.patch), self.parameters.nbins, weights=self.kernel)
        hist_sum = sum(hist)
        self.q = np.array([i / hist_sum for i in hist])     # Normalized histogram

    def track(self, image):

        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)

        right = min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1)

        if right - left < self.patch.shape[1] or bottom - top < self.patch.shape[0]:
            return [self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0], self.size[1]], 0
        
        # Calculate new positions of the patch using mean shift algorithm
        new_position, iter_count, _ = mean_shift(image, self.kernel, self.position, self.parameters.nbins, self.q, self.size, self.parameters.eps)

        # Update current patch
        self.patch, _ = get_patch(image, new_position, self.size)

        # Update q
        hist = extract_histogram(reshape_patch(self.patch), self.parameters.nbins, weights=self.kernel)
        hist_sum = sum(hist)
        normalized_hist = np.array([i / hist_sum for i in hist])
        self.q = (1 - self.parameters.alpha) * self.q + self.parameters.alpha * normalized_hist
        
        # Update position values
        self.position = new_position

        # Return the values
        return [int(new_position[0] - self.size[0] // 2), int(new_position[1] - self.size[1] // 2), self.size[0], self.size[1]], iter_count


class MSParams():
    def __init__(self):
        self.enlarge_factor = 2
        self.nbins = 16
        self.eps = 1
        self.sigma = 0.5
        self.alpha = 0.5
