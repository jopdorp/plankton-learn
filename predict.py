import theano
import numpy as np
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
import pandas



def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]
    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), y_true.T.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss

def predict(model_path, batch_size):
	model = serial.load(model_path)
	dataset = yaml_parse.load(model.dataset_yaml_src)
	dataset = dataset.get_test_set()

	model.set_batch_size(batch_size)
	# dataset must be multiple of batch size of some batches will have
	# different sizes. theano convolution requires a hard-coded batch size
	m = dataset.X.shape[0]
	extra = batch_size - m % batch_size
	assert (m + extra) % batch_size == 0
	import numpy as np
	if extra > 0:
	    dataset.X = np.concatenate((dataset.X, np.zeros((extra, dataset.X.shape[1]),
	    dtype=dataset.X.dtype)), axis=0)
	assert dataset.X.shape[0] % batch_size == 0


	X = model.get_input_space().make_batch_theano()
	Y = model.fprop(X)

	from theano import function

	f = function([X], Y)


	y = []

	for i in xrange(dataset.X.shape[0] / batch_size):
	    x_arg = dataset.X[i*batch_size:(i+1)*batch_size,:]
	    if X.ndim > 2:
	        x_arg = dataset.get_topological_view(x_arg)
	    y.append(f(x_arg.astype(X.dtype)))

	y = np.concatenate(y)
	assert y.shape[0] == dataset.X.shape[0]
	# discard any zero-padding that was used to give the batches uniform size
	return dataset.y,y[:m]

def create_co_occurence_matrix(y_actual,y_predicted):
	data_frame = pandas.DataFrame()
	data_frame['predicted'] = y_predicted
	data_frame['actual'] = y_actual
	data_frame['count'] = np.ones(len(y_predicted))
	dummy_df = pandas.DataFrame()
	dummy_df['predicted'] = sorted(list(set(y_actual) | set(y_predicted)))
	dummy_df['actual'] = data_frame['predicted'] 
	dummy_df['count'] = np.zeros(len(dummy_df['actual']))
	data_frame = pandas.concat([data_frame, dummy_df])
	return data_frame.pivot_table(index='predicted',
		columns='actual',
		values='count',
		aggfunc=np.sum
	).fillna(0)

def never_predicted_classes(y_predicted,n_classes):
	return set(range(n_classes)) - set(y_predicted)