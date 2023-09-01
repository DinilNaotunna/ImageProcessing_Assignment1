import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from google.colab.patches import cv2_imshow 

# Load the image
img = cv.imread('shells.tif',cv.IMREAD_GRAYSCALE)

#equalized image
equalized_img = cv.equalizeHist(img)

def plot_histograms(img):
      # Calculate the histograms of the grayscale image and the equalized image
    hist_img = cv.calcHist([img], [0], None, [256], [0, 256])
    equ = cv.equalizeHist(img)
    hist_equ = cv.calcHist([equ], [0], None, [256], [0, 256])

    # Plot the histograms side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(hist_img), axs[0].set_title('Histogram of Grayscale Image')
    axs[1].plot(hist_equ),axs[1].set_title('Histogram of Equalized Image')
    plt.show()
# Display the histograms before and after equalization
plot_histograms(img)

cv2_imshow( img)
cv2_imshow( equalized_img )
cv.waitKey(0)
cv.destroyAllWindows()