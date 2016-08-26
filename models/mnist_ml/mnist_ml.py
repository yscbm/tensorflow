# -*- coding:utf-8 -*-  

from sys import path

import tensorflow as tf
import numpy as np
path.append('../..')
from common import extract_mnist
from common import tf_log

#初始化单个卷积核上的参数
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def main():
	#创建图
	#输入数据，28*28的图片，placeholder为占位符，等同于声明了没赋值
	x = tf.placeholder("float", [None, 784])

	#神经网络的权值
	#W = tf.Variable(tf.zeros([784,10]))
	W = weight_variable([784,10])
	#偏置量
	b = bias_variable([10])

	saver = tf.train.Saver()

	#图片[784]*权值[784,10]+偏置[10],获得该图片在每个类（0~9）的价值，用softmax，选择最大值的类
	y = tf.nn.softmax(tf.matmul(x,W) + b)

	#补丁，防止y等于0，造成log(y)计算出-inf
	y1 = tf.maximum(y,1e-30)
	#正确的值
	y_ = tf.placeholder("float", [None,10])

	#交叉熵代价函数 C = -1/n * sum(y_*ln(y)+(1-y_)*ln(1-y))
	cross_entropy = -tf.reduce_sum(y_*tf.log(y1))

	#调用梯度下降的优化算法，自动反向传播误差更新权值，偏置量，还有AdagradOptimizer等优化算法
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

	#初始化所有变量
	init = tf.initialize_all_variables()

	tflog = tf_log.tfLog()


	#开始进行计算
	sess = tf.InteractiveSession()

	#调用初始化
	sess.run(init)

	#实例化MnistDataSet类，在common/extract_mnist.py中声明，用来得到mnist的数据，imageType为区别图片维度（28*28还是784）
	mnist_data_set = extract_mnist.MnistDataSet('../../data/')

	test_images,test_labels = mnist_data_set.test_data()

	#调用800次batch
	for i in xrange(100000):
		#获取一个批次的数据
		batch_xs, batch_ys = mnist_data_set.next_train_batch(100)

		#把数据传入图中（之前有占位），然后运行这个图
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
		#rr = sess.run([train_step,y,y1,b], feed_dict={x: batch_xs, y_: batch_ys})
		#print i
		#print rr[1][0]
		#print rr[2][0]
		#print rr[3]


		if i%1000 == 0:
			correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
			print sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})

	#保存模型参数
	if not tf.gfile.Exists('model_data'):
		tf.gfile.MakeDirs('model_data')
	save_path = saver.save(sess, "model_data/model.ckpt")
	print "Model saved in file: ", save_path
	#测试准确程度
	#对比预测值与真实值是否一致
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	
	#求正确率
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	
	#运行并输出结果
	print sess.run(accuracy, feed_dict={x: test_images, y_: test_labels})

	#关闭会话
	sess.close()

if __name__ == '__main__':
	main()