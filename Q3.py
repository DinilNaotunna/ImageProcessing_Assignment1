import cv2
import matplotlib.pyplot as plt
import numpy as np
# Load the input image
input_image = cv2.imread('highlights_and_shadows.jpg', cv2.IMREAD_COLOR)

# Set the gamma value
gamma_value = 1

# Convert BGR image to Lab color space
lab_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2Lab)

# Split the Lab image into channels
L_channel, a_channel, b_channel = cv2.split(lab_image)

# Calculate gamma correction lookup table
gamma_table = np.array([(i / 255.0) ** (gamma_value) * 255.0 for i in np.arange(0, 256)]).astype('uint8')

# Apply gamma correction to the L channel
corrected_L_channel = cv2.LUT(L_channel, gamma_table)

# Merge the corrected L channel with a and b channels
corrected_lab_image = cv2.merge((corrected_L_channel, a_channel, b_channel))

# Convert corrected Lab image back to BGR color space
gamma_corrected_image = cv2.cvtColor(corrected_lab_image, cv2.COLOR_Lab2BGR)

# Create subplots for displaying images and histograms
fig, axarr = plt.subplots(2, 2, figsize=(10, 8))

# Display the original image
axarr[0, 0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
axarr[0, 0].set_title('Original Image')
axarr[0, 0].axis('off')

# Display the gamma-corrected image
axarr[0, 1].imshow(cv2.cvtColor(gamma_corrected_image, cv2.COLOR_BGR2RGB))
axarr[0, 1].set_title('Gamma Corrected Image')
axarr[0, 1].axis('off')

# Plot histograms for original and gamma-corrected images
colors = ('b', 'g', 'r')
for i, c in enumerate(colors):
    hist_input = cv2.calcHist([input_image], [i], None, [256], [0, 256])
    axarr[1, 0].plot(hist_input, color=c)
    hist_corrected = cv2.calcHist([gamma_corrected_image], [i], None, [256], [0, 256])
    axarr[1, 1].plot(hist_corrected, color=c)

# Display the subplots
plt.tight_layout()
plt.show()
