import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
input_image = cv.imread('flower.jpg')
# (a)
segmentation_mask = np.zeros(input_image.shape[:2], dtype=np.uint8)

# Define a region of interest (ROI) within the image
roi_rect = (30, 30, input_image.shape[1] - 30, input_image.shape[0] - 150)
bg_model = np.zeros((1, 65), dtype=np.float64)# Initialize arrays for background and foreground models
fg_model = np.zeros((1, 65), dtype=np.float64)
cv.grabCut(input_image, segmentation_mask, roi_rect, bg_model, fg_model, 5, cv.GC_INIT_WITH_RECT)
# Create a binary mask where 1 represents the foreground and 0 represents the background
binary_mask = np.where((segmentation_mask == 2) | (segmentation_mask == 0), 0, 1).astype('uint8')
# Apply the mask to the original image to extract the segmented foreground
segmented_image = input_image * binary_mask[:, :, np.newaxis]

# Create subplots for visualization
fig, ax = plt.subplots(1, 5, figsize=(15, 15))
ax[0].imshow(binary_mask, cmap='gray'),ax[0].set_title('Segmentation Mask')
ax[1].imshow(cv.cvtColor(segmented_image, cv.COLOR_BGR2RGB)),ax[1].set_title('Segmented Foreground')
ax[2].imshow(cv.cvtColor(input_image - segmented_image, cv.COLOR_BGR2RGB)),ax[2].set_title('Segmented Background')

# (b) Enhance the image
blurred_background = cv.GaussianBlur(input_image, (0, 0), 30)
enhanced_image = np.where(binary_mask[:, :, np.newaxis] == 1, input_image, blurred_background)

ax[3].imshow(cv.cvtColor(input_image, cv.COLOR_BGR2RGB)),ax[3].set_title('Original Image')
ax[4].imshow(cv.cvtColor(enhanced_image, cv.COLOR_BGR2RGB)),ax[4].set_title('Enhanced Image')

plt.show()

