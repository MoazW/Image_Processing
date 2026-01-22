import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import img_as_float, img_as_ubyte
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio
from skimage.util import random_noise
from skimage.color import rgb2gray
import os

#from scipy import ndimage as nd #ndimage can also be used
from skimage.morphology import opening, closing, square, 

folder_path = "/image/"
filename = "BSE_Image.jpg"
path=os.path.join(folder_path,filename)


img = img_as_float(io.imread(path))

# estimate the noise standard deviation from the noisy image
sigma_est = np.mean(estimate_sigma(img,channel_axis=-1))
print(f'estimated noise standard deviation = {sigma_est}')

patch_kw = dict(
    patch_size=5,  # 5x5 patches
    patch_distance=6,  # 13x13 search area
    channel_axis=-1,
)

#Denoise
denoise = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True, **patch_kw)

#Show histogram to determine segments
denoise_ubyte = img_as_ubyte(rgb2gray(denoise))
#plt.hist(denoise_ubyte.flat, bins =200), range=(0,255)


#Segmentation
seg1 = (denoise_ubyte <=55)
seg2= (denoise_ubyte > 55) & (denoise_ubyte <=110)
seg3= (denoise_ubyte > 110) & (denoise_ubyte <= 210)
seg4= (denoise_ubyte >210)
all_segments = np.zeros((denoise_ubyte.shape[0],denoise_ubyte.shape[1],3))
all_segments[seg1, :]=[1,0,0]
all_segments[seg2, :]= [0,1,0]
all_segments[seg3,:]= [0,0,1]
all_segments[seg4,:] = [1,1,0]


#plt.imshow(all_segments)
#plt.show()



seg1_opened = opening(seg1, square(3))
seg1_closed = closing(seg1_opened, square(3))

seg2_opened = opening(seg2, square(3))
seg2_closed = closing(seg2_opened, square(3))

seg3_opened = opening(seg3, square(3))
seg3_closed = closing(seg3_opened, square(3))

seg4_opened = opening(seg4, square(3))
seg4_closed = closing(seg4_opened, square(3))

all_segments_cleaned = np.zeros((denoise_ubyte.shape[0],denoise_ubyte.shape[1],3))

all_segments_cleaned[seg1_closed, :]=[1,0,0]
all_segments_cleaned[seg2_closed, :]= [0,1,0]
all_segments_cleaned[seg3_closed,:]= [0,0,1]
all_segments_cleaned[seg4_closed,:] = [1,1,0]


#plt.imshow(all_segments_cleaned)
#plt.show()
plt.imsave(path+"segmented.jpg",all_segments)
plt.imsave(path+"segmented_final.jpg",all_segments_cleaned)

credits: @Sreeni
