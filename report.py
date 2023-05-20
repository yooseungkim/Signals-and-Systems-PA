import matplotlib.pyplot as plt
import numpy as np
from implementations import *


def task1(img: np.array, show=True):  # for task1 : fourier transform and visualization
    fourier = fft2(img)  # 2D discrete Fourier transform
    shifted = np.fft.fftshift(fourier)  # centerize
    if show:
        plt.imshow(np.log(np.abs(shifted)), cmap="gray")  # visualize
        plt.show()
    return shifted  # return fourier transformed (+ shifted) image


def task2(images):  # for task2 : azimuthal averaging
    azs = []  # 1D power spectrums
    for i in range(len(images)):
        fft_img = np.log(np.abs(np.fft.fftshift(fft2(images[i]))))
        az = azimuthal_averaging(fft_img)
        azs.append(az)
        plt.plot(az)
    # plotting
    plt.title("1D Power Spectrum")
    plt.xlabel("Spatial Frequency")
    plt.ylabel("Power Spectrum")
    # plt.ylim(0.25, 1.2)
    # plt.xlim(-20, 250)
    plt.legend([f"00{i}.jpg" for i in range(7)])
    plt.show()
    return azs


def task3(img: np.array, show=True):  # for task 3: high-pass filtering * inverse fourier transform
    fft_img = fft2(img)  # fourier transform
    fft_shifted = np.fft.fftshift(fft_img)  # shift
    highpass_img = highpass(fft_shifted)  # high-pass filtering
    inv_fft_img = inverse_fft2(highpass_img)  # inverse fourier transform
    img = np.abs(inv_fft_img)  # convert to real numbers
    if show:
        plt.imshow(img, cmap="gray")  # visualize
        plt.show()
    return img


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


def get_center(means, epoch=10):
    centers = tuple(np.random.random(2))  # initial centers, choose randomly
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
    print(centers)
    return centers


def detect(images, show=True):
    means = []
    for i in range(len(images)):
        # original inverse fourier transformed image
        inv_image = task3(images[i], show=False)
        eye_cropped = inv_image[eye_location[i]]  # cropped eye image
        if show:
            plt.imshow(eye_cropped, cmap="gray")  # visualize
            plt.show()
        # compute mean, mean value of image will be used as an criterion
        means.append(eye_cropped.mean())

    # get center of two clusters, real and fake
    centers = get_center(means)
    # classify image into two clusters that is closer to its mean
    predicted = [True if abs(centers[0] - m) >
                 abs(centers[1] - m) else False for m in means]
    print(predicted)
    return predicted


# examples
task1(images[0])
task2(images)
task3(images[0])
detect(images, show=False)
