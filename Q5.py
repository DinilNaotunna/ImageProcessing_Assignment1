import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv.imread('shells.tif',cv.IMREAD_COLOR)



def plot_histograms(img):
    # Convert the image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Calculate the histograms of the grayscale image and the equalized image
    hist_gray = cv.calcHist([gray], [0], None, [256], [0, 256])
    equ = cv.equalizeHist(gray)
    hist_equ = cv.calcHist([equ], [0], None, [256], [0, 256])

    # Plot the histograms side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(hist_gray), axs[0].set_title('Histogram of Grayscale Image')
    axs[1].plot(hist_equ),axs[1].set_title('Histogram of Equalized Image')
    plt.show()

# Display the histograms before and after equalization
plot_histograms(img)