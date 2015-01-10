import plankton_utils as pu
import random as rnd

import numpy as np

from pylearn2.datasets import control
from pylearn2.datasets import dense_design_matrix

TEST_TRAIN_RANDOM_SEED=12345
TEST_SIZE=6067
TRAIN_SIZE=24269
IM_SIZE=32
IM_GLOB="converted_images/*" 

class PlanktonDDM(dense_design_matrix.DenseDesignMatrix):

    def __init__(self, which_set, start=None, stop=None):

        self.args = locals()                                                     
                                                                                 
        if which_set not in ['train', 'test']:                                   
            raise ValueError(                                                    
                'Unrecognized which_set value "%s".' % (which_set,) +            
                '". Valid values are ["train","test"].')

        size = TEST_SIZE if which_set == 'test' else TRAIN_SIZE
        if control.get_load_data():
            topo_view, y = self.read_images(size,
                                            True if which_set == 'train' else False) 
        else:
            topo_view = np.random.rand(size, 32, 32)
            y = np.random.randint(0, 10, (size, 1))

        super(PlanktonDDM, self).__init__(topo_view=topo_view, y=y,
                                    axes=['b', 0, 1, 'c'],
                                    y_labels=len(np.unique(y)))

        if start is not None:
            assert start >= 0
            if stop > self.X.shape[0]:
                raise ValueError('stop=' + str(stop) + '>' +
                                 'm=' + str(self.X.shape[0]))
            assert stop > start
            self.X = self.X[start:stop, :]
            if self.X.shape[0] != stop - start:
                raise ValueError("X.shape[0]: %d. start: %d stop: %d"
                                 % (self.X.shape[0], start, stop))
            if len(self.y.shape) > 1:
                self.y = self.y[start:stop, :]
            else:
                self.y = self.y[start:stop]
            assert self.y.shape[0] == stop - start


    def read_images(self, size, take_from_end):
        classes = pu.get_classes(IM_GLOB)
         
        imlab = list(pu.images(pu.train_image_paths(IM_GLOB)))
        rnd.seed(TEST_TRAIN_RANDOM_SEED) 
        rnd.shuffle(imlab)
        if take_from_end:
          imlab = imlab[-size:]
        else:
          imlab = imlab[:size]
           
        images = np.array([i for _, i in imlab])[:,:,:,np.newaxis]
        labels = np.array([classes[l] for l, _ in imlab])[:,np.newaxis]
        
        return images, labels
         
