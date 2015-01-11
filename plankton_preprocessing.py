import os
import logging
import shutil
from theano import config
from pylearn2.datasets import preprocessing
from plankton_design_matrix import PlanktonDDM
from pylearn2.utils.string_utils import preprocess

orig_path = preprocess('../train')
train_name ='h5/splitted_train_32x32.h5'
valid_name = 'h5/valid_32x32.h5'
test_name = 'h5/test_32x32.h5'

# # copy data if don't exist
# if not os.path.isdir(os.path.join("../train", 'h5')):
#     os.makedirs(os.path.join("../train", 'h5'))

# for d_set in [train_name, valid_name, test_name]:
#     if not os.path.isfile(os.path.join("../train", d_set)):
#         logging.info("Copying data from {0} to {1}".format(os.path.join("../train", d_set), "../train"))
#         shutil.copyfile(os.path.join(orig_path, d_set),
#                     os.path.join("../train", d_set))

def check_dtype(data):
    if str(data.X.dtype) != config.floatX:
        logging.warning("The dataset is saved as {}, changing theano's floatX "\
                "to the same dtype".format(data.X.dtype))
        config.floatX = str(data.X.dtype)

# Load train data
train = PlanktonDDM('train')
check_dtype(train)

# prepare preprocessing
pipeline = preprocessing.Pipeline()
# without batch_size there is a high chance that you might encounter memory error
# or pytables crashes
pipeline.items.append(preprocessing.GlobalContrastNormalization())
pipeline.items.append(preprocessing.ZCA((32,32)))

# apply the preprocessings to train
train.apply_preprocessor(pipeline, can_fit=True)
del train

# load and preprocess test
test = PlanktonDDM('test')
check_dtype(test)
test.apply_preprocessor(pipeline, can_fit=False)
