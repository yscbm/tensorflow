# -*- coding:utf-8 -*-  

import tensorflow as tf

def main():
	#创建图
	#输入数据，28*28的图片，placeholder为占位符，等同于声明了没赋值
	x = tf.placeholder("float", [None, 784])

	#神经网络的权值
	W = tf.Variable(tf.zeros([784,10]))

	#偏置量
	b = tf.Variable(tf.zeros([10]))

	#图片[784]*权值[784,10]+偏置[10],获得该图片在每个类（0~9）的价值，用softmax，选择最大值的类
	y = tf.nn.softmax(tf.matmul(x,W) + b)

	#正确的值
	y_ = tf.placeholder("float", [None,10])

	#交叉熵代价函数 C = -1/n * sum(y_*ln(y)+(1-y_)*ln(1-y))
	cross_entropy = -tf.reduce_sum(y_*tf.log(y))

	#调用梯度下降的优化算法，自动反向传播误差更新权值，偏置量，还有AdagradOptimizer等优化算法
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

	#初始化所有变量
	init = tf.initialize_all_variables()

	#开始进行计算
	sess = tf.Session()
	
	#调用初始化
	sess.run(init)




if __name__ == '__main__':
	main()