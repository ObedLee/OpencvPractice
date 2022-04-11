# 필요한 패키지를 import함
from __future__ import print_function
import argparse
import cv2
import numpy as np
from handle_channel_roi import display_image

def smoothing_image(img, type='mean', ksize=3):
	# type에 따라 평활화 방법 선택
	if(type == 'mean'):
		dst = cv2.blur(img, (ksize,ksize))
	elif(type == 'Gaussian'):
		dst = cv2.GaussianBlur(img, (ksize,ksize), 0)
	elif(type == 'median'):
		dst = cv2.medianBlur(img, ksize)
	else:
		print('[에러]잘못된 type 지정')
		return None

	return dst

def gradient_image(img, type='Sobel', dir='all'):
	# type에 따라 경계 강도 계산 방법 선택
	if(type == 'Sobel'):
		if(dir == 'x'):
			dst = cv2.Sobel(image, cv2.CV_64F, 1, 0)
			dst = np.uint8(np.absolute(dst))
		elif(dir == 'y'):
			dst = cv2.Sobel(image, cv2.CV_64F, 0, 1)
			dst = np.uint8(np.absolute(dst))
		elif(dir == 'all'):
			dstx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
			dstx = cv2.convertScaleAbs(dstx)
			dsty = cv2.Sobel(image, cv2.CV_64F, 0, 1)
			dsty = cv2.convertScaleAbs(dsty)
			dst = cv2.addWeighted(dstx, 1, dsty, 1, 0)
	elif(type == 'Laplacian'):
		dst = cv2.Laplacian(img, cv2.CV_64F)
		dst = np.uint8(np.absolute(dst))
	else:
		print('[에러]잘못된 type 지정')
		return None

	return dst

def edge_detection(img, lth=100, hth=300):
	if len(img.shape) == 3:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	dst = cv2.Canny(img, lth, hth)

	return dst

def filtering_image(img, kernel=None):
	if(kernel is None):
		kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
	dst = cv2.filter2D(img, -1, kernel)

	return dst

if __name__ == '__main__' :
	# 명령행 인자 처리
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--image', required = True, \
			help = 'Path to the input image')
	args = vars(ap.parse_args())

	filename = args['image']

	# OpenCV를 사용하여 영상 데이터 로딩
	image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
	display_image(image, 'original', False)

	smoothing = smoothing_image(image, 'Gaussian', 5)
	if(smoothing is not None):
		display_image(smoothing, 'smoothing', False)

	gradient = gradient_image(image, 'Laplacian', 'all')
	if(gradient is not None):
		display_image(gradient, 'gradient', False)

	edge = edge_detection(image)
	if(edge is not None):
		display_image(edge, 'edge detection', False)

	filtering = filtering_image(image)
	if(filtering is not None):
		display_image(filtering, 'filtering')