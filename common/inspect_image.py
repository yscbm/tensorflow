# -*- coding:utf-8 -*-  

import cv2

def save_image(image,name = 'test'):
	cv2.imwrite(name, image)

def save_image_name(image,name):
	#cv2.imwrite(name, image)
	print name,image
def main():
	pass

if __name__ == '__main__':
	main()