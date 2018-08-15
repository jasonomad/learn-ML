import util.config as cfg
import os
import scipy.io as sio
import tensorflow as tf
import numpy as np
import util.config as cfg

# import cv2

def load_ship(is_training=True):
	path = os.path.join('data', 'ship')
	if is_training:
        # load data
		tr_data = sio.loadmat(cfg.TRAIN_FILE)
		val_data = sio.loadmat(cfg.VAL_FILE)
		
        # retrieve shapes from dataset
		num_val = val_data['val_x'].shape[0]
		num_tr = tr_data['train_x'].shape[0]
		
        # extract training set and validation set 
        # reshape -> [batch, row, col, ch]
		trX = tr_data['train_x'].reshape(num_tr,
				cfg.IMG_INPUT_SIZE, cfg.IMG_INPUT_SIZE, cfg.IMG_INPUT_CHANNEL).astype(np.float32)/256
		trY = np.zeros([num_tr, cfg.NUM_CLASS]) # one-hot coding
		trY[:,tr_data['train_y']] = 1

		valX = val_data['val_x'].reshape((num_val,
				cfg.IMG_INPUT_SIZE, cfg.IMG_INPUT_SIZE, cfg.IMG_INPUT_CHANNEL)).astype(np.float32)/256
		valY = np.zeros([num_val, cfg.NUM_CLASS]) # one-hot coding
		valY[:,val_data['val_y']] = 1
        
		'''
		# debug
		cv2.imshow("Image", trX[123] + 1)
		cv2.waitKey(0)
		#
		'''
		
		return trX, trY, valX, valY
	else:
		te_data = sio.loadmat(cfg.TEST_FILE)
		num_te = te_data['test_x'].shape[0]
		teX = te_data['test_x'].reshape((num_te,
				cfg.IMG_INPUT_SIZE, cfg.IMG_INPUT_SIZE, cfg.IMG_INPUT_CHANNEL)).astype(np.float32)/256
		teY = np.zeros([num_te, cfg.NUM_CLASS]) # one-hot coding
		teY[:,te_data['test_y']] = 1
        
		return teX, teY


def get_batch(batch_size, data_x, data_y):

	n_sample = data_x.shape[0]
	
	assert(batch_size <= n_sample)
	idx = np.arange(n_sample)
	
	# np.random.seed(1)
	np.random.shuffle(idx)
	
	'''
	data_queues = tf.train.slice_input_producer([trX, trY])
	X, Y = tf.train.shuffle_batch(data_queues,
									batch_size = batch_size,
									allow_smaller_final_batch = False,
									capacity=batch_size * 64,
									min_after_dequeue=batch_size * 32)
	'''
	
	data_x_batch = data_x[idx[0:batch_size]]
	data_y_batch = data_y[idx[0:batch_size]]
	
	return data_x_batch, data_y_batch   