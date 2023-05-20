import matplotlib.pyplot as plt
import numpy as np
from fourier import *


def task1(img: np.array):  # for task1 : fourier transform and visualization
    fourier = fft2(img)  # 2D discrete Fourier transform
    shifted = np.fft.fftshift(fourier)  # centerize
    plt.imshow(np.log(np.abs(shifted)), cmap="gray")  # visualize
    plt.show()
    return shifted  # return fourier transformed (+ shifted) image


def task2():  # for task2 : azimuthal averaging
    azs = []  # 1D power spectrums
    for i in range(7):
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


def task3(img: np.array):  # for task 3: high-pass filtering * inverse fourier transform
    fft_img = fft2(img)  # fourier transform
    fft_shifted = np.fft.fftshift(fft_img)  # shift
    highpass_img = highpass(fft_shifted)  # high-pass filtering
    inv_fft_img = inverse_fft2(highpass_img)  # inverse fourier transform
    img = np.abs(inv_fft_img)  # convert to real numbers
    plt.imshow(img, cmap="gray")  # visualize
    plt.show()
    return img


task3(images[0])
