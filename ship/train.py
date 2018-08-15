import tensorflow as tf
import os
import datetime
import numpy as np

from network import Network
import util.config as cfg
from util.ship_data import load_ship, get_batch
 
slim = tf.contrib.slim

class Solver(object):
		
	def __init__(self, net):
		# retreive parameters from config
		self.net = net
		self.batch_size = cfg.BATCH_SIZE
		self.print_iter = cfg.PRINT_ITER
		self.save_iter = cfg.SAVE_ITER
		self.max_iter = cfg.MAX_ITER
		self.initial_learning_rate = cfg.LEARNING_RATE
		self.decay_steps = cfg.DECAY_STEP
		self.decay_rate = cfg.LEARNING_RATE_DECAY
		self.staircase = cfg.STAIRCASE
		self.validate_iter = cfg.VALIDATE_ITER
		self.summary_iter = cfg.SUMMARY_STEP
		
        # tf 
		self.global_step = tf.train.create_global_step()
		self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate,
							self.global_step, self.decay_steps, self.decay_rate, self.staircase, name='learning_rate')
		self.optimizer = tf.train.AdamOptimizer(
									learning_rate=self.learning_rate)
		self.train_op = slim.learning.create_train_op(
						self.net.total_loss, self.optimizer, global_step=self.global_step)
		
		# create a session
		self.sess = tf.Session()
		# initialize
		self.sess.run(tf.global_variables_initializer())
		
		# saving directory and file name
		self.output_dir = cfg.OUTPUT_DIR
		self.saver = tf.train.Saver(tf.global_variables())
		self.ckpt_file = os.path.join(self.output_dir, 'ConvNet')
		
		# add summary
		tf.summary.scalar('total_train_loss', self.net.total_loss)
		tf.summary.scalar('train_accuracy_batch', self.net.accuracy)
		
		self.merged = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter("logs/", self.sess.graph)
		
	def train(self, train_x, train_y, val_x, val_y):
	
	
		num_sample = train_x.shape[0]
		# batch
		for step in range(1, self.max_iter + 1):
		
			# get a batch
			batch_x, batch_y = get_batch(self.batch_size, train_x, train_y)
			
			feed_dict = {self.net.input_batch: batch_x,
							self.net.labels: batch_y}
							
			if step % self.print_iter == 0:
				loss, _, acc, mrgd = self.sess.run([self.net.total_loss, self.train_op, self.net.accuracy, self.merged],
										feed_dict = feed_dict)
				
				# add summary
				#tf.summary.scalar('total_train_loss', loss)
				#tf.summary.scalar('train_accuracy_batch', acc)				
				
				print('step = ', step,
						', epoch = ', step*self.batch_size//num_sample,
						', loss = ', loss,
						', accuracy this batch =', acc)
						
			else:
				_, mrgd = self.sess.run([self.train_op, self.merged],
									feed_dict = feed_dict)
									
			if step % self.validate_iter == 0:
				val_acc = self.evaluate(val_x, val_y)
				print('step = ', step,
						', accuracy on validation set =', val_acc)
						
				# add summary				
				#tf.summary.scalar('validation_accuracy', val_acc)
			
			# save checkpoint
			if step % self.save_iter == 0:
				print('saving checkpoint file')
				self.saver.save(self.sess,
								self.ckpt_file,
								global_step=self.global_step)
			
			# log
			if step % self.summary_iter == 0:
				self.writer.add_summary(mrgd, step)
								
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
			
		total_acc = total_correct/num_sample
		
		return total_acc
			
			
		
								
								
def main():
    
	# build network
    convnet = Network()
	# load data
    train_x, train_y, val_x, val_y = load_ship(is_training = True)
    # create solver
    solver = Solver(convnet)
    # training
    print('Start training ...')
    solver.train(train_x, train_y, val_x, val_y)
    print('Done training.')


if __name__ == '__main__':
    main()
                