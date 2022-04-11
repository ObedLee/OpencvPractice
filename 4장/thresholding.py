# 필요한 패키지를 import함
from __future__ import print_function
import argparse
import cv2
from matplotlib import pyplot as plt

def thresholding_image(img, thresh=-1):
	if thresh < 0:
		option = cv2.THRESH_BINARY + cv2.THRESH_OTSU
	else:
		option = cv2.THRESH_BINARY

	th, dst = cv2.threshold(img, thresh, 255, option)

	return th, dst

if __name__ == "__main__" :
	# 명령행 인자 처리
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--image', required = True, \
			help = 'Path to the input image')
	args = vars(ap.parse_args())

	filename = args['image']

	# OpenCV를 사용하여 영상 데이터 로딩
	image = cv2.imread(filename)

	# Grayscale 영상으로 변환
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# thresholding 수행
	th1, dst1 = thresholding_image(gray, 127)
	print('global threshold (without blurring): ', th1)

	# blurring 적용후 Otsu thresholding 수행
	blur = cv2.GaussianBlur(gray, (5,5), 0)
	th2, dst2 = thresholding_image(blur)
	print('optimal threshold (after blurring) with Otsu: ', th2)

	# thresholding 결과 표시
	plt.subplot(2, 3, 1), plt.imshow(gray, 'gray')
	plt.title('original'), plt.xticks([]), plt.yticks([])
	plt.subplot(2, 3, 2)
	plt.hist(gray.ravel(), 256, density=True, color='k')
	plt.title('histogram'), plt.axis([0, 255, 0, 0.02]), plt.yticks([])
	plt.subplot(2, 3, 3), plt.imshow(dst1, 'gray')
	plt.title('thresholding(127)'), plt.xticks([]), plt.yticks([])

	plt.subplot(2, 3, 4), plt.imshow(blur, 'gray')
	plt.title('blurring'), plt.xticks([]), plt.yticks([])
	plt.subplot(2, 3, 5), plt.hist(blur.ravel(), 256)
	plt.title('histogram'), plt.xlim([0, 256]), plt.yticks([])
	plt.subplot(2, 3, 6), plt.imshow(dst2, 'gray')
	plt.title('Otsu'), plt.xticks([]), plt.yticks([])

	plt.show() 