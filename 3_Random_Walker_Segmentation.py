from skimage import io, img_as_float, exposure
from skimage.transform import resize
import matplotlib.pyplot as plt
import os
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.morphology import opening, closing, square
import numpy as np
from skimage.segmentation import random_walker


    

folder = "/Image/"
filename= "NiCrAlloy.jpg"

path = os.path.join(folder,filename)

img= img_as_float(io.imread(path))
print(img.shape)
cropped_img=img[:250,:]

def nlm(image,patch_size, patch_distance,channel_axis):
    sigma_est = np.mean(estimate_sigma(image, channel_axis=channel_axis))
    denoise_img=denoise_nl_means(
        image,
        h=1.15*sigma_est,
        fast_mode=True,
        patch_size=patch_size, patch_distance=patch_distance,
        channel_axis=channel_axis
    )
    return denoise_img

denoise_img=nlm(cropped_img,5,6,None)



eq_img = exposure.equalize_adapthist(denoise_img)

markers = np.zeros(cropped_img.shape,dtype=np.uint)
markers[(eq_img<0.4)]=1
markers[(eq_img>0.7)]=2

labels = random_walker(eq_img, markers, beta=5, mode='bf')

seg1 = (labels == 1)
seg2 = (labels ==2)
seg1_opened = opening(seg1, square(3))
seg1_closed = closing(seg1_opened, square(3))

seg2_opened = opening(seg2, square(3))
seg2_closed = closing(seg2_opened, square(3))

allsegments = np.zeros((eq_img.shape[0], eq_img.shape[1],3))
allsegments[seg1_closed]=(1,0,0)
allsegments[seg2_closed]=(0,1,0)


fig, axes = plt.subplots(
    2, 2,
    constrained_layout=True
    #figsize=(10,10)
)
axes[0,0].imshow(cropped_img, cmap = 'gray')
axes[0,0].set_title("Original")
axes[0,0].axis('off')
axes[0,1].imshow(markers, cmap = 'gray')
axes[0,1].set_title("Markers")
axes[0,1].axis('off')
axes[1,0].imshow(labels, cmap = 'gray')
axes[1,0].set_title("Random Walker Segmentation")
axes[1,0].axis('off')
axes[1,1].imshow(allsegments)
axes[1,1].set_title("Segmented")
axes[1,1].axis('off')
plt.savefig(folder+"Walker.jpg", dpi=300, bbox_inches="tight" )

#credits: @Sreeni

