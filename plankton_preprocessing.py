import os

import plankton_design_matrix as pdm
from pylearn2.datasets import preprocessing                                      
from pylearn2.utils import serial                                                


OUTPUT_DIR="zca_images"

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
    
