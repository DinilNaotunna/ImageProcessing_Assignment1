import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Load the image
img = cv.imread('jeniffer.jpg')

# (a)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
hue, sat, val = cv.split(hsv)

fig, ax = plt.subplots(1, 3, figsize=(14, 14))
ax[0].imshow(hue, cmap='gray'),ax[0].set_title("Hue plane")
ax[1].imshow(sat, cmap='gray'),ax[1].set_title("Saturation plane")
ax[2].imshow(val, cmap='gray'),ax[2].set_title("Value plane")
plt.show()

# (b) using the saturation (s) plane for thresholding
_, thresholded_mask = cv.threshold(sat, 110, 255, cv.THRESH_BINARY)

# (c)
foreground = cv.bitwise_and(val, thresholded_mask)

# (d)
hist_foreground = cv.calcHist([foreground], [0], thresholded_mask, [256], [0, 256])
cumulative_hist = np.cumsum(hist_foreground)

# (e)
equalized_foreground = cv.equalizeHist(foreground)

# (f)
background = cv.bitwise_not(thresholded_mask)
background_image = cv.bitwise_and(val, background)

# Add the equalized foreground with the background
result_value = cv.add(equalized_foreground, background_image)

# Display results
fig, ax = plt.subplots(1, 3, figsize=(14, 14))
ax[0].imshow(thresholded_mask, cmap='gray')
ax[0].set_title("Thresholded Mask")
ax[1].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
ax[1].set_title("Original")

ax[2].imshow(result_value, cmap='gray')
ax[2].set_title("Result with Histogram-Equalized Foreground")
plt.show()
