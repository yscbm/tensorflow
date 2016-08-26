# -*- coding:utf-8 -*-  

from sys import path

import tensorflow as tf
import time

path.append('../..')
from common import extract_cifar10
from common import inspect_image


#初始化单个卷积核上的参数
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

#初始化单个卷积核上的偏置值
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

#卷积操作
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')



def main():
	#定义会话
	sess = tf.InteractiveSession()
	
	#声明输入图片数据，类别
	x = tf.placeholder('float',[None,32,32,3])
	y_ = tf.placeholder('float',[None,10])

	#第一层卷积层
	W_conv1 = weight_variable([5, 5, 3, 64])
	b_conv1 = bias_variable([64])
	#进行卷积操作，并添加relu激活函数
	conv1 = tf.nn.relu(conv2d(x,W_conv1) + b_conv1)
	# pool1
	pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool1')
	# norm1
	norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')


	#第二层卷积层
	W_conv2 = weight_variable([5,5,64,64])
	b_conv2 = bias_variable([64])
	conv2 = tf.nn.relu(conv2d(norm1,W_conv2) + b_conv2)
	# norm2
	norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')
	# pool2
	pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME', name='pool2')

	#全连接层
	#权值参数
	W_fc1 = weight_variable([8*8*64,384])
	#偏置值
	b_fc1 = bias_variable([384])
	#将卷积的产出展开
	pool2_flat = tf.reshape(pool2,[-1,8*8*64])
	#神经网络计算，并添加relu激活函数
	fc1 = tf.nn.relu(tf.matmul(pool2_flat,W_fc1) + b_fc1)
	
	#全连接第二层
	#权值参数
	W_fc2 = weight_variable([384,192])
	#偏置值
	b_fc2 = bias_variable([192])
	#神经网络计算，并添加relu激活函数
	fc2 = tf.nn.relu(tf.matmul(fc1,W_fc2) + b_fc2)


	#Dropout层，可控制是否有一定几率的神经元失效，防止过拟合，训练时使用，测试时不使用
	keep_prob = tf.placeholder("float")
	#Dropout计算
	fc1_drop = tf.nn.dropout(fc2,keep_prob)
	
	#输出层，使用softmax进行多分类
	W_fc2 = weight_variable([192,10])
	b_fc2 = bias_variable([10])
	y_conv=tf.maximum(tf.nn.softmax(tf.matmul(fc1_drop, W_fc2) + b_fc2),1e-30)

	#补丁，防止y等于0，造成log(y)计算出-inf
	#y1 = tf.maximum(y_conv,1e-30)

	#代价函数
	cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
	#使用Adam优化算法来调整参数
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	
	#测试正确率
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

	#保存模型训练数据
	saver = tf.train.Saver()

	#所有变量进行初始化
	sess.run(tf.initialize_all_variables())

	#获取cifar10数据
	cifar10_data_set = extract_cifar10.Cifar10DataSet('../../data/')
	test_images,test_labels = cifar10_data_set.test_data()

	#进行训练
	start_time = time.time()
	for i in xrange(20000):
		#获取训练数据
		#print i,'1'
		batch_xs, batch_ys = cifar10_data_set.next_train_batch(50)
		#print i,'2'

		#每迭代100个 batch，对当前训练数据进行测试，输出当前预测准确率
		if i%1000 == 0:
			#print "test accuracy %g"%accuracy.eval(feed_dict={x: test_images, y_: test_labels, keep_prob: 1.0})
			train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 1.0})
			print "step %d, training accuracy %g"%(i, train_accuracy)
			#计算间隔时间
			end_time = time.time()
			print 'time: ',(end_time - start_time)
			start_time = end_time


		if (i+1)%10000 == 0:
			#输出整体测试数据的情况
			avg = 0
			for j in xrange(20):
				avg+=accuracy.eval(feed_dict={x: test_images[j*50:j*50+50], y_: test_labels[j*50:j*50+50], keep_prob: 1.0})
			avg/=20
			print "test accuracy %g"%avg
			#保存模型参数
			if not tf.gfile.Exists('model_data'):
				tf.gfile.MakeDirs('model_data')
			save_path = saver.save(sess, "model_data/model.ckpt")
			print "Model saved in file: ", save_path



		train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

	#输出整体测试数据的情况
	avg = 0
	for i in xrange(200):
		avg+=accuracy.eval(feed_dict={x: test_images[i*50:i*50+50], y_: test_labels[i*50:i*50+50], keep_prob: 1.0})
	avg/=200
	print "test accuracy %g"%avg


	#关闭会话
	sess.close()

if __name__ == '__main__':
	main()