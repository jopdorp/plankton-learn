import glob
import os

import numpy as np

from skimage import morphology
from skimage import measure
from skimage.io import imread
from skimage.transform import resize

BASE='../train/*'

def classes(base_glob=BASE):
    return [d.split(os.sep)[-1] for d in glob.glob(base_glob)] 

def train_image_paths(base_glob=BASE):
    for cls_path in glob.glob(base_glob):
        cls = cls_path.split(os.sep)[-1]
        for impath in image_paths(cls_path):
            yield cls, impath 

def image_paths(path):
    for path, folder, files in os.walk(path): 
        for f in files:                                                      
            if f[-4:] == ".jpg":                                             
                yield "{0}{1}{2}".format(path, os.sep, f)

def images(impaths):
   for cls, impath in impaths:                                                    
       image = imread(impath, as_grey=True)
       yield cls, image
               
def scale_images(images, imsize_x=25, imsize_y=25):
   for cls, image in images:
       yield cls, resize(image, (imsize_x, imsize_y))

def attach_ratio(images):
    for cls, image in images:
        yield cls, image, min_max_region_ratio(image) 

def min_max_region_ratio(image):
    image = image.copy()
    # Create the thresholded image to eliminate some of the background
    imagethr = np.where(image > np.mean(image),0.,1.0)

    #Dilate the image
    imdilated = morphology.dilation(imagethr, np.ones((4,4)))
    label_list = measure.label(imdilated)
    label_list = imagethr*label_list
    label_list = label_list.astype(int)
    region_list = measure.regionprops(label_list)
    maxregion = getLargestRegion(region_list, label_list, imagethr)
            
    # guard against cases where the segmentation fails by providing zeros
    ratio = 0.0
    if ((not maxregion is None) and  (maxregion.major_axis_length != 0.0)):
        ratio = maxregion.minor_axis_length*1.0 / maxregion.major_axis_length
    return ratio

def getLargestRegion(props, labelmap, imagethres):
    regionmaxprop = None
    for regionprop in props:
        # check to see if the region is at least 50% nonzero
        if sum(imagethres[labelmap == regionprop.label])*1.0/regionprop.area < 0.50:
            continue
        if regionmaxprop is None:
            regionmaxprop = regionprop
        if regionmaxprop.filled_area < regionprop.filled_area:
            regionmaxprop = regionprop
    return regionmaxprop

