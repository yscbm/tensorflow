# -*- coding:utf-8 -*-  

from sys import path


import tensorflow as tf

path.append('../..')
from common import extract_mnist


#初始化单个卷积核上的参数
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

#初始化单个卷积核上的偏置值
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

#输入特征x，用卷积核W进行卷积运算，strides为卷积核移动步长，
#padding表示是否需要补齐边缘像素使输出图像大小不变
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#对x进行最大池化操作，ksize进行池化的范围，
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


def main():
	
	#定义会话
	sess = tf.InteractiveSession()
	
	#声明输入图片数据，类别
	x = tf.placeholder('float',[None,784])
	y_ = tf.placeholder('float',[None,10])
	#输入图片数据转化
	x_image = tf.reshape(x,[-1,28,28,1])

	#第一层卷积层，初始化卷积核参数、偏置值，该卷积层5*5大小，一个通道，共有32个不同卷积核
	#[filter_height, filter_width, in_channels, out_channels]
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])
	#进行卷积操作，并添加relu激活函数
	h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
	#进行最大池化
	h_pool1 = max_pool_2x2(h_conv1)

	#同理第二层卷积层
	W_conv2 = weight_variable([5,5,32,64])
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)
	
	#全连接层
	#权值参数
	W_fc1 = weight_variable([7*7*64,1024])
	#偏置值
	b_fc1 = bias_variable([1024])
	#将卷积的产出展开
	h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
	#神经网络计算，并添加relu激活函数
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
	
	#Dropout层，可控制是否有一定几率的神经元失效，防止过拟合，训练时使用，测试时不使用
	keep_prob = tf.placeholder("float")
	#Dropout计算
	h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
	
	#输出层，使用softmax进行多分类
	W_fc2 = weight_variable([1024,10])
	b_fc2 = bias_variable([10])
	y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


	#代价函数
	cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
	#使用Adam优化算法来调整参数
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	
	#测试正确率
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	
	#保存参数
	saver = tf.train.Saver()

	#所有变量进行初始化
	sess.run(tf.initialize_all_variables())

	#获取mnist数据
	mnist_data_set = extract_mnist.MnistDataSet('../../data/')
	test_images,test_labels = mnist_data_set.test_data()

	#进行训练
	for i in xrange(20000):
		#获取训练数据
		batch_xs, batch_ys = mnist_data_set.next_train_batch(100)

		#每迭代100个 batch，对当前训练数据进行测试，输出当前预测准确率
		if i%100 == 0:
			train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 1.0})
			print "step %d, training accuracy %g"%(i, train_accuracy)
		
		#训练数据
		train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
	
	#保存参数
	save_path = saver.save(sess, "model_data/model.ckpt")
	print "Model saved in file: ", save_path

	#输出整体测试数据的情况
	print "test accuracy %g"%accuracy.eval(feed_dict={x: test_images, y_: test_labels, keep_prob: 1.0})

	#关闭会话
	sess.close()


if __name__ == '__main__':
	main()

