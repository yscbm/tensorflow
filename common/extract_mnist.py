# -*- coding:utf-8 -*-  

import gzip

'''
TRAINING SET IMAGE FILE (train-images-idx3-ubyte):

[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000803(2051) magic number 
0004     32 bit integer  60000            number of images 
0008     32 bit integer  28               number of rows 
0012     32 bit integer  28               number of columns 
0016     unsigned byte   ??               pixel 


TRAINING SET LABEL FILE (train-labels-idx1-ubyte):

[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
0004     32 bit integer  60000            number of items 
0008     unsigned byte   ??               label 


'''
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255

TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

TRAIN_NUM = 80000
TEST_NUM = 10000

def extract_images(filename, num_images):

	with gzip.open(filename) as bytestream:
		#按上面格式去除没用的数据
		bytestream.read(16)
		#读取数据
		buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
		#把数据流转化为numpy的数组
		data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
		#数据从0~255转为-0.5~0.5
		#data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
		#原数据为1维数组，转换为4维，每个维度分别对应下面
		#data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)

		return data



def extract_labels(filename, num_images):

	with gzip.open(filename) as bytestream:
		#按上面格式去除没用的数据
		bytestream.read(8)
		#读取数据
		buf = bytestream.read(1 * num_images)
		#把数据流转化为numpy的数组
		labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)

		return labels

class MnistDataSet(object):
	"""docstring for MnistDataSet"""
	def __init__(self,data_dir,imageType = '784'):
		super(MnistDataSet, self).__init__()
		#mnist数据集分用于卷积神经网络和BP神经网络
		self.imageType = imageType
		#提取数据
		self.train_images =  extract_images(data_dir+TRAIN_IMAGES,TRAIN_NUM)
		self.train_labels = extract_labels(data_dir+TRAIN_LABELS,TRAIN_NUM)
		self.test_images = extract_images(data_dir+TEST_IMAGES,TEST_NUM)
		self.test_labels = extract_labels(data_dir+TEST_LABELS,TEST_NUM)
		#epoch为整体数据迭代一次
		#batch为用来计算梯度调整参数的一个批次
		#epoch完成次数
		self.epochs_completed = 0
		#当前批次在epoch中进行的进度
		self.index_in_epoch = 0

		#根据imageType划分数据
		if imageType == '784':
			self.train_images = self.train_images.reshape(TRAIN_LABELS, IMAGE_SIZE*IMAGE_SIZE, NUM_CHANNELS)
			self.test_images = self.test_images.reshape(TEST_NUM, IMAGE_SIZE*IMAGE_SIZE, NUM_CHANNELS)
		else:
			self.train_images = self.train_images.reshape(TRAIN_LABELS, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
			self.test_images = self.test_images.reshape(TEST_NUM, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)

	def next_train_batch(self,batch_size):
		#起始位置
		start = self.index_in_epoch
		self.index_in_epoch += batch_size
		#完成了一次epoch
		if self.index_in_epoch > TRAIN_NUM:
			#epoch完成次数加1
			self.epochs_completed += 1
			#打乱数据顺序，随机梯度下降
			perm = numpy.arange(TRAIN_NUM)
			numpy.random.shuffle(perm)
			self.images = self.images[perm]
			self.labels = self.labels[perm]
			start = 0
			self.index_in_epoch = batch_size
			#条件不成立会报错
			assert batch_size <= TRAIN_NUM
		end = self.index_in_epoch
		return self.images[start:end], self.labels[start:end]

#def next_batch_784()