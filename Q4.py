import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('/spider.png',cv2.IMREAD_COLOR)

# Convert image from RGB to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Split image into hue, saturation, and value planes #(a)
h, s, v = cv2.split(hsv)

# Apply intensity transformation to saturation plane  #(b) & (c)
a = 0.45 # adjust this value to get a visually pleasing output
s = np.minimum(s + a*128*np.exp(-(s-128)**2/(2*70**2)), 255).astype(np.uint8)

# Recombine the three planes #(d)
hsv = cv2.merge((h, s, v))

# Convert image back to RGB color space
result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# Plot original and transformed images #(e)
fig, ax = plt.subplots(1,3,figsize=(12,12))
ax[0].imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB)), ax[0].axis('off'), ax[0].set_title("Original image")
ax[1].imshow(cv.cvtColor(result,cv.COLOR_BGR2RGB)), ax[1].axis('off'), ax[1].set_title("Vibrance enhanced image")
##ax[2].imshow(s), ax[2].axis('off'), ax[2].set_title("Intensity transformation")
ax[2].plot(s), ax[2].set_title("Intensity transformation"), ax[2].set_aspect('equal')
plt.show()
