# -*- coding:utf-8 -*-  
from sys import path
import tarfile
import os


import tensorflow as tf

path.append('..')
from common import download

SOURCE_URL = 'http://www.cs.toronto.edu/~kriz/'
WORK_DIRECTORY = '../data/cifar10'
FILE_LIST = ('cifar-10-binary.tar.gz',)


def main():
	print "You can use other methods to download:"
	print "http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
	for filename in FILE_LIST:
		download.download_file(WORK_DIRECTORY,filename,SOURCE_URL)
		filepath = os.path.join(WORK_DIRECTORY,filename)

		tarfile.open(filepath, 'r:gz').extractall(WORK_DIRECTORY)

	print "Successfully downloaded"
if __name__ == '__main__':
	main()





