# 필요한 패키지를 import함
from __future__ import print_function
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt

def histogram_slicing(img, lowerb, upperb):
	# 주어진 범위 내부의 값으로만 구성되는 마스크 생성
	# 픽셀값이 주어진 범위 내부인 경우 마스크는 True 아니면 False로 설정
	mask = cv2.inRange(img, lowerb, upperb)

	# Bitwise-AND mask and original image
	dst = cv2.bitwise_and(img, img, mask = mask)

	return dst

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

	# Grayscale 영상에서 범위를 설정하여 객체 분할
	lower = 80
	upper = 170
	blurg = cv2.GaussianBlur(gray, (5,5), 0)
	dst1 = histogram_slicing(blurg, lower, upper)

	# HSV 색상 모델로 변환
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	# HSV 영상에서 녹색 범위를 설정하여 객체 분할
	lowerb = np.array([50, 50, 50])
	upperb = np.array([70, 255, 255])
	dst2 = histogram_slicing(blurhsv, lowerb, upperb)

	# RGB 색상 모델로 변환
	dst2 = cv2.cvtColor(dst2, cv2.COLOR_HSV2RGB)

	# 객체 분할 결과 표시
	plt.subplot(2, 3, 1), plt.imshow(gray, 'gray')
	plt.title('original'), plt.xticks([]), plt.yticks([])
	plt.subplot(2, 3, 2)
	plt.hist(blurg.ravel(), 256, density=True, color='k')
	plt.plot([80, 80], [0, 0.02], [170, 170], [0, 0.02])
	plt.title('histogram'), plt.axis([0, 255, 0, 0.02]), plt.yticks([])
	plt.subplot(2, 3, 3), plt.imshow(dst1, 'gray')
	plt.title('histogram_slicing'), plt.xticks([]), plt.yticks([])

	plt.subplot(2, 3, 4), plt.imshow(image)
	plt.title('original'), plt.xticks([]), plt.yticks([])
	plt.subplot(2, 3, 5)
	plt.hist(hsv[:,:,0].ravel(), 180, density=True, color='k')
	plt.plot([50, 50], [0, 0.02], [70, 70], [0, 0.02])
	plt.title('histogram'), plt.axis([0, 180, 0, 0.02]), plt.yticks([])
	plt.subplot(2, 3, 6), plt.imshow(dst2)
	plt.title('histogram_slicing'), plt.xticks([]), plt.yticks([])

	plt.show()