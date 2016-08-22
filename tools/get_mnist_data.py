# -*- coding:utf-8 -*-  

from sys import path

import tensorflow as tf

path.append('../common')
import download

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = '../data/mnist'
FILE_LIST = ('train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz')

def main():
	for filename in FILE_LIST:
		download.download_file(WORK_DIRECTORY,filename,SOURCE_URL)

if __name__ == '__main__':
	main()