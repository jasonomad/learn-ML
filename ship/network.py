import numpy as np
import tensorflow as tf
import util.config as cfg

slim = tf.contrib.slim

class Network(object):


	def __init__(self):
	
		self.img_size = cfg.IMG_INPUT_SIZE
		self.num_class = cfg.NUM_CLASS
		# self.random_seed = cfg.RANDOM_SEED
		
		self.input_batch = tf.placeholder(tf.float32, 
                                          [None, self.img_size, self.img_size, 1],
                                          name = 'input')
										  
		self.logits = self.build_network(self.input_batch,
                                         num_outputs = self.num_class)
        
		self.labels = tf.placeholder(tf.float32,
                                     [None, self.num_class])
        
		self.loss_layer(self.logits, self.labels)
		
		self.total_loss = slim.losses.get_total_loss()
		
		self.acc_num = self.evaluate_acc(self.logits, self.labels)
		
		self.accuracy = self.acc_num/cfg.BATCH_SIZE
		
	def build_network(self,
					input_batch,
					num_outputs,
					scope='convnet'):
		
		with tf.variable_scope(scope):
			with slim.arg_scope([slim.conv2d, slim.fully_connected],
					normalizer_fn = slim.batch_norm,
                    weights_regularizer = slim.l2_regularizer(0.0001),
                    weights_initializer = tf.glorot_normal_initializer()):
					
				net = input_batch
				
				net = slim.conv2d(net, 8, 3, 2, padding='SAME', activation_fn = tf.nn.relu, scope='conv_1')
				net = slim.conv2d(net, 8, 3, 2, padding='SAME', activation_fn = tf.nn.relu, scope='conv_2')
				net = slim.max_pool2d(net, kernel_size = 4, stride = 4, padding='SAME', scope='pool_3')
				
				net = slim.conv2d(net, 8, 3, 2, padding='SAME', activation_fn = tf.nn.relu, scope='conv_4')
				net = slim.conv2d(net, 8, 3, 2, padding='SAME', activation_fn = tf.nn.relu, scope='conv_5')
				net = slim.max_pool2d(net, kernel_size = 4, stride = 4, padding='SAME', scope='pool_6')
				
				net = slim.conv2d(net, 16, 3, 2, padding='SAME', activation_fn = tf.nn.relu, scope='conv_7')
				net = slim.conv2d(net, 16, 3, 2, padding='SAME', activation_fn = tf.nn.relu, scope='conv_8')
				net = slim.max_pool2d(net, kernel_size = 2, stride = 2, padding='SAME', scope='pool_9')
				
				net = slim.conv2d(net, 16, 3, 2, padding='SAME', activation_fn = tf.nn.relu, scope='conv_10')
				net = slim.conv2d(net, 16, 3, 2, padding='SAME', activation_fn = tf.nn.relu, scope='conv_11')
				net = slim.max_pool2d(net, kernel_size = 2, stride = 2, padding='SAME', scope='pool_12')
				
				net = slim.flatten(net, scope='flat_17')
				net = slim.fully_connected(net, 4, activation_fn = tf.nn.relu, scope='fc_14')
				#net = slim.fully_connected(net, 16, activation_fn = tf.nn.relu, scope='fc_15')
				net = slim.fully_connected(net, num_outputs, activation_fn=None, scope='fc_16')
				
		return net
		
	def loss_layer(self, predicts, labels, scope='loss_layer'):
		with tf.variable_scope(scope):
			class_loss = slim.losses.sigmoid_cross_entropy(logits = predicts, multi_class_labels = labels)
			slim.losses.add_loss(class_loss)
			
	def evaluate_acc(self, predicts, labels, scope='eva_acc'):
		with tf.variable_scope(scope):
			argmax_predict = tf.argmax(predicts, axis = 1)
			argmax_label = tf.argmax(labels, axis = 1)
			correct_prediction = tf.equal(argmax_predict, argmax_label)
			correct_num = tf.reduce_sum(tf.cast(correct_prediction, dtype = tf.float32))
			
		return correct_num
            
        