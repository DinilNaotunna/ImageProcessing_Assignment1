#Q8
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

original_images = ["im02small.png", "im03small.png", "im09small.png","im11small.png"]
zoom_outs = ["im02.png", "im03.png", "im09.png", "im11.png"]

def images_set():
 for j in range(4):
    image = cv.imread(original_images[j])
    image_zoom_out = cv.imread(zoom_outs[j])

    image_bilinear = cv.resize(image, None, fx=4, fy=4, interpolation=cv.INTER_LINEAR)
    image_near = cv.resize(image, None, fx=4, fy=4, interpolation=cv.INTER_NEAREST)
    
    fig, ax = plt.subplots(1,4, figsize=(15,15))
    ax[0].imshow(cv.cvtColor(image,cv.COLOR_BGR2RGB)), ax[0].set_title("Original image")
    ax[1].imshow(cv.cvtColor(image_near,cv.COLOR_BGR2RGB)), ax[1].set_title("Nearest-neighbor zoomed image")
    ax[2].imshow(cv.cvtColor(image_bilinear,cv.COLOR_BGR2RGB)), ax[2].set_title("Bilinear interpolation zoomed image")
    ax[3].imshow(cv.cvtColor(image_zoom_out ,cv.COLOR_BGR2RGB)), ax[3].set_title("Zoomed-out version")
    plt.show()

images_set()
