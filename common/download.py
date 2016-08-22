# -*- coding:utf-8 -*-  

import gzip
import os
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange  
import tensorflow as tf

#下载状况
def report_hook(count, block_size, total_size):
	percent = 100.0 * count * block_size/ total_size
	sys.stdout.write("\r%02d%%" % int(percent<=100 and percent or 100) + ' complete')
	sys.stdout.flush()

def download_file(dirpath,filename,url):
	#判断文件路径是否存在
	if not tf.gfile.Exists(dirpath):
		tf.gfile.MakeDirs(dirpath)
	filepath = os.path.join(dirpath, filename)
	#判断文件是否存在、需要下载
	if not tf.gfile.Exists(filepath):
		print filename
		filepath, _ = urllib.request.urlretrieve(url + filename, filepath,reporthook= report_hook)
		with tf.gfile.GFile(filepath) as f:
			size = f.Size()
		print('Successfully downloaded', filename, size, 'bytes.')
	return filepath
