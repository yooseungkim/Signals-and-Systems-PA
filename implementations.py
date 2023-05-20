import cv2
import matplotlib.pyplot as plt
import numpy as np
images = [cv2.imread("./images/data/00{}.jpg".format(i),
                     cv2.IMREAD_GRAYSCALE) for i in range(0, 7)]

# 2D discrete Fourier transform


def fft2(img: np.array):
    M, N = img.shape
    fourier = np.matmul(exp_mu(img), np.matmul(img, exp_nv(img)))
    return fourier

# 2D discrete inverse Fourier transform


def inverse_fft2(img: np.array):
    M, N = img.shape
    inv_fourier = np.matmul(exp_mu(img, inverse=True), np.matmul(
        img, exp_nv(img, inverse=True))) / (M * N)
    return inv_fourier

# high-pass filtering


def highpass(img, radius=45):
    h, w = img.shape[0], img.shape[1]
    filtered = [[0 for _ in range(w)] for _ in range(h)]
    for i in range(h):
        for j in range(w):
            distance = ((i - h // 2) ** 2 + (j - w // 2) ** 2)
            if distance < radius ** 2:
                continue
            filtered[i][j] = img[i][j]
    return np.array(filtered)

# for 2D discrete Fourier transform


def exp_mu(img: np.array, inverse=False):
    M = img.shape[0]
    result = [[0 for _ in range(M)] for _ in range(M)]
    for u in range(M):
        for m in range(M):
            if inverse:
                result[u][m] = np.exp(2j * np.pi * m * u / M)
            else:
                result[u][m] = np.exp(-2j * np.pi * m * u / M)
    return np.array(result)


def exp_nv(img: np.array, inverse=True):
    N = img.shape[1]
    result = [[0 for _ in range(N)] for _ in range(N)]
    for v in range(N):
        for n in range(N):
            if inverse:
                result[v][n] = np.exp(2j * np.pi * n * v / N)
            else:
                result[v][n] = np.exp(-2j * np.pi * n * v / N)
    return np.array(result)

# azimuthal averaging


def azimuthal_averaging(img: np.ndarray):
    h, w = img.shape
    center = (h // 2, w // 2)
    # max radius (distance from the center to the corner)
    max_radius = np.hypot(center[0], center[1])
    if max_radius != int(max_radius):  # if max_radius is not integer
        max_radius += 1
    max_radius = int(max_radius)
    cum_sum_freq = np.array([0 for _ in range(max_radius)])
    pixels = np.array([0 for _ in range(max_radius)])
    for i in range(h):
        for j in range(w):
            # distance from the center
            radius = int(np.hypot(i - center[0], j - center[1]))
            cum_sum_freq[radius] += img[i][j]
            pixels[radius] += 1
    cum_sum_freq = cum_sum_freq / pixels  # divide into the number of pixels
    cum_sum_freq = cum_sum_freq / max(cum_sum_freq)  # averaging
    return cum_sum_freq


def get_center(means, epoch=10):
    # initial centers, choose randomly, but 0 is for fake 1 is for real, so centers[0] < centers[1]
    centers = tuple(np.random.random(2).sort())
    for epoch in range(10):
        # denote which cluster each image belongs to
        assigned = [0 for _ in range(7)]
        sums = [0, 0]  # to compute new centers (average )
        for i, m in enumerate(means):
            if abs(centers[0] - m) > abs(centers[1] - m):
                # assign to the cluster with closer center, in this case, it is closer to 1
                assigned[i] = 1
                sums[1] += m
            else:
                # assigned to 0 by default
                sums[0] += m
        # compute new centers with assigned images
        new_centers = (sums[0] / (assigned.count(0) + 1e-100),
                       sums[1] / (assigned.count(1) + 1e-100))
        # add small value to avoid devision by zero error

        # update new centers until it converges
        if centers == new_centers:
            break
        centers = new_centers
    return centers


# for fake detection
# eye location for each image
eye_location = loc = [
    np.ix_(range(70, 150), range(60, 260)),
    np.ix_(range(120, 160), range(0, 160)),
    np.ix_(range(80, 120), range(75, 225)),
    np.ix_(range(80, 130), range(75, 255)),
    np.ix_(range(80, 140), range(75, 255)),
    np.ix_(range(80, 140), range(75, 255)),
    np.ix_(range(80, 140), range(70, 230)),
]
