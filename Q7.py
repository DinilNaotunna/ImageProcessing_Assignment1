#Q7(a)
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('einstein.png', cv.IMREAD_GRAYSCALE).astype(np.float32)
# Sobel kernels
sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float32')

# Apply Sobel kernels using filter2D
img_sobel_h = cv.filter2D(img, -1, sobel_kernel_x)
img_sobel_v = cv.filter2D(img, -1, sobel_kernel_y)

# Display the images
fig, axes = plt.subplots(1, 3, figsize=(18, 9))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original')
axes[1].imshow(img_sobel_h, cmap='gray')
axes[1].set_title('Sobel Horizontal')
axes[2].imshow(img_sobel_v, cmap='gray')
axes[2].set_title('Sobel Vertical')
plt.show()



#Q7(b)
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np 

img = cv.imread("einstein.png", cv.IMREAD_GRAYSCALE).astype(np.float32)

# Sobel vertical edge detection
img_sobel_v = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)

# Sobel horizontal edge detection
img_sobel_h = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)

fig, axes = plt.subplots(1, 3, figsize=(18, 9))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original')
axes[1].imshow(img_sobel_v, cmap='gray')
axes[1].set_title('Sobel Vertical')
axes[2].imshow(img_sobel_h, cmap='gray')
axes[2].set_title('Sobel Horizontal')
plt.show()



#Q7 (c)
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("einstein.png", cv.IMREAD_GRAYSCALE).astype(np.float32)

k_hor = np.array([1,2,1], dtype=np.float32)
k_ver = np.array([1,0,-1], dtype=np.float32)
img_sobel_ver = cv.sepFilter2D(img, -1, k_hor, k_ver)
img_sobel_hor = cv.sepFilter2D(img, -1, k_ver, k_hor)

fig, ax  = plt.subplots(1, 3, figsize=(18,9))
ax[0].imshow(img, cmap='gray'), ax[0].set_title('Original')
ax[1].imshow(img_sobel_v, cmap='gray'), ax[1].set_title('Sobel Vertical')
ax[2].imshow(img_sobel_h, cmap='gray'), ax[2].set_title('Sobel Horizontal')
plt.show()

