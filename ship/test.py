import tensorflow as tf
import os
import datetime
import numpy as np

from network import Network
import util.config as cfg
from util.ship_data import load_ship, get_batch
 
slim = tf.contrib.slim

class Classifier(object):
		
	def __init__(self, net, weight_dir):
		# retreive parameters from config
		self.net = net
		
		self.weight_dir = weight_dir
		
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		
		print('Restoring Weights from Directory: ' + self.weight_dir)
		self.saver = tf.train.Saver()
		self.ckpt = tf.train.get_checkpoint_state(self.weight_dir)
		if self.ckpt and self.ckpt.model_checkpoint_path:
			self.saver.restore(self.sess, self.ckpt.model_checkpoint_path)
		else:
			print('No Weight File or DIR')
	

	def test(self, test_x, test_y):
	
		test_acc = self.evaluate(test_x, test_y)
		
		print('Accuracy on test set is: ', test_acc)
	
	
	def evaluate(self, data_x, data_y):
	
		num_sample = data_x.shape[0]
		batch_size = cfg.BATCH_SIZE
		batch_num = int(np.ceil(num_sample/batch_size))
		
		total_correct = 0
		for step in range(batch_num):
			if step == batch_num - 1:
				x_batch = data_x[step*batch_size::]
				y_batch = data_y[step*batch_size::]
			else:
				x_batch = data_x[step*batch_size:(step+1)*batch_size]
				y_batch = data_y[step*batch_size:(step+1)*batch_size]
			
			feed_dict = {self.net.input_batch: x_batch,
							self.net.labels: y_batch}
			correct_num = self.sess.run([self.net.acc_num], feed_dict = feed_dict)
			total_correct += correct_num[0]
			# print(correct_num[0], ' / ', x_batch.shape[0])
		
		print(int(total_correct), ' / ', num_sample)
		total_acc = total_correct/num_sample
		
		return total_acc
		
def main():

	# build network
    convnet = Network()
	# load data
    test_x, test_y = load_ship(is_training = False)
    # create solver
    classifier = Classifier(convnet, cfg.WEIGHT_DIR)
    # testing
    print('Start testing ...')
    classifier.test(test_x, test_y)
    print('Done testing.')
	
if __name__ == '__main__':
	main()