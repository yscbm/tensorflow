# -*- coding:utf-8 -*-  

import os

class tfLog(object):
	"""docstring for tfLog"""
	def __init__(self, logDir=''):
		super(tfLog, self).__init__()
		self.logDir = logDir
	
	def saveLog(self,v):
		data=open(os.path.join(self.logDir,'log'),'a')
		data.write(v) 
		data.close() 


def main():
	tt = tfLog('')
	tt.saveLog('123')

if __name__ == '__main__':
	main()