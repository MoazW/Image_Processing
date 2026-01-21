
from skimage import io
from skimage.filters.rank import entropy
from skimage.filters import threshold_otsu
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.stats import linregress
from skimage.morphology import disk
from skimage.color import rgb2gray
from pathlib import Path
from skimage.util import img_as_ubyte

time_list=[]
area_list=[]
time=0

folderpath = "/image/assay/*.*"
folder= sorted(glob.glob(folderpath))
for image in folder:
    img = io.imread(image)
    if img.ndim == 3:  # check if image is RGB
        img = rgb2gray(img)  # returns float in [0,1]
    img = img_as_ubyte(img) # returns uint8 needed for entrop
    entropy_img = entropy(img,disk(5))
    thresh = threshold_otsu(entropy_img)
    binary = entropy_img <= thresh
    scratch_area = np.sum( binary ==1)
    print("time=", time, "hr ", "Scratch area=", scratch_area, "pix\N{SUPERSCRIPT TWO}" )
    time_list.append(time)
    area_list.append(scratch_area)
    time+=1
    p = Path(image)
    plt.imsave(p.with_name(f"Entropy_{p.stem}_analysis.jpg"),entropy_img)
    #io.imsave(p.with_name(f"Entropy_{p.stem}_analysis.tif"), binary.astype(np.float32))

#plt.plot(time_list, area_list)
#plt.show()
#print(entropy_img.min(), entropy_img.max())
#slope, intercept, r_value, p_value, std_err = linregress(time_list, scratch_area)
#print("y=", slope, "x", "+", intercept)
#print("R\N{SUPERSCRIPT TWO} = ", r_value**2)
#Credits: @DigitalSreeni
