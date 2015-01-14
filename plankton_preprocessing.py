import os

import plankton_utils as pu
from skimage.transform import resize, rotate

import plankton_design_matrix as pdm
from pylearn2.datasets import preprocessing                                      
from pylearn2.utils import serial                                                
from scipy.misc import imsave

CONVERTED_IMAGES_DIR="converted_images"
OUTPUT_DIR="zca_images"

def convert_image(im):
    #oriented = resize(pu.orientate_image(im), (32, 32))
    #rot90 = rotate(oriented, 90, resize=True, cval=1.0)
    #return oriented, rot90
    return (resize(im, (32, 32)),)

if not os.path.exists(CONVERTED_IMAGES_DIR):
    os.makedirs(CONVERTED_IMAGES_DIR)
    images = pu.images(pu.train_image_paths())
    for i, im in enumerate(images):
        out_dir = os.path.join(CONVERTED_IMAGES_DIR, im[0])
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for j, cim in enumerate(convert_image(im[1])):
            imsave(os.path.join(out_dir, "%d_%d.png" %(i,j)), cim)
else:
    print "not converting images bcs %s exists" %(CONVERTED_IMAGES_DIR,) 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

    train = pdm.PlanktonDDM(which_set = 'train')
    test = pdm.PlanktonDDM(which_set = 'test')  

    preprocessor = preprocessing.ZCA()                                               
    train.apply_preprocessor(preprocessor = preprocessor, can_fit = True)            
    test.apply_preprocessor(preprocessor = preprocessor, can_fit = False) 


    train.use_design_loc(OUTPUT_DIR + '/train.npy')                                    
    serial.save(OUTPUT_DIR + '/train.pkl', train)  

    test.use_design_loc(OUTPUT_DIR + '/test.npy')                                      
    serial.save(OUTPUT_DIR + '/test.pkl', test)

    serial.save(OUTPUT_DIR + '/preprocessor.pkl',preprocessor)
else:
    print "output directory already exist"
    
