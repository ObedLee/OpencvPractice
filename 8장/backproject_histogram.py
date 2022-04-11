# 필요한 패키지를 import함
from __future__ import print_function
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt

def back_project(img_ref, img_tar):
	r_hsv = cv2.cvtColor(img_ref, cv2.COLOR_BGR2HSV)
	t_hsv = cv2.cvtColor(img_tar, cv2.COLOR_BGR2HSV)

	# calculating histogram for reference image
	r_hist = cv2.calcHist([r_hsv], [0,1], None, \
		 	 [180,256],  [0,180, 0,256] )

	# apply backprojection
	dst = cv2.calcBackProject([t_hsv], [0,1], r_hist, [0,180,0,256], 1)

	# convolute with circular disc
	ele = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
	cv2.filter2D(dst, -1, ele, dst)

	return dst

if __name__ == "__main__" :
	# 명령행 인자 처리
	ap = argparse.ArgumentParser()
	ap.add_argument('-r', '--reference', required = True, \
			help = 'Path to the reference image')
	ap.add_argument('-t', '--target', required = True, \
			help = 'Path to the target image')
	args = vars(ap.parse_args())

	f_ref = args['reference']
	f_tar = args['target']

	# OpenCV를 사용하여 영상 데이터 로딩
	img_ref = cv2.imread(f_ref)
	img_tar = cv2.imread(f_tar)

	# excute histogram backprojection
	proj = back_project(img_ref, img_tar)

	# threshold and binary AND
	th, th_proj = cv2.threshold(proj, 50, 255, 0)
	th_proj = cv2.merge((th_proj, th_proj, th_proj))

	img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
	img_tar = cv2.cvtColor(img_tar, cv2.COLOR_BGR2RGB)

	# bitwise and operation for generating result
	r_img = cv2.bitwise_and(img_tar, th_proj)

	plt.subplot(2, 2, 1), plt.imshow(img_tar)
	plt.title('target'), plt.xticks([]), plt.yticks([])
	plt.subplot(2, 2, 2), plt.imshow(img_ref)
	plt.title('reference'), plt.xticks([]), plt.yticks([])
	plt.subplot(2, 2, 3), plt.imshow(proj, 'gray')
	plt.title('probability'), plt.xticks([]), plt.yticks([])
	plt.subplot(2, 2, 4), plt.imshow(r_img)
	plt.title('bitwise_and'), plt.xticks([]), plt.yticks([])
	plt.show()