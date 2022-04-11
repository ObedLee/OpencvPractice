# 필요한 패키지를 import함
from __future__ import print_function
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt

def histogram(img):

	# 그레이레벨 히스토그램 계산
	hist = cv2.calcHist([img], [0], None, [256], [0, 256])

	# 누적 히스토그램 생성
	cdf = hist.cumsum()
	# 빈도를 0 ~ 1사이로 정규화
	cdf = cdf * float(max(hist)) / max(cdf)

	# 결과 히스토그램 반환
	return hist, cdf

if __name__ == "__main__" :
	# 명령행 인자 처리
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--image', required = True, \
			help = 'Path to the input image')
	ap.add_argument('-t', '--type', type = int, \
			default = 1,
			help = 'type of histogram equalize(1: original, 2: CLAHE)')
	args = vars(ap.parse_args())

	filename = args['image']
	type = args['type']

	# OpenCV를 사용하여 영상 데이터 로딩
	image = cv2.imread(filename)

	# 그레이스케일 영상으로 변환
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# 원본 영상에 대한 히스토그램 계산
	hist1, cdf1 = histogram(image)

	# 원본 영상에 대한 히스토그램 출력
	plt.subplot(2, 2, 1), plt.imshow(image, 'gray')
	plt.title('image'), plt.xticks([]), plt.yticks([])
	plt.subplot(2, 2, 2)
	plt.plot(hist1, color='k')
	plt.plot(cdf1, color='r')
	plt.title('histogram'), plt.xlim([0,256])

	# 히스토그램 균일화 수행
	if type == 1:
		equalize = cv2.equalizeHist(image)
	elif type == 2:
		clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
		equalize = clahe.apply(image)

	# 히스토그램 균일화 영상에 대한 히스토그램 계산
	hist2, cdf2 = histogram(equalize)

	# 히스토그램 균일화 영상에 대한 히스토그램 출력
	plt.subplot(2, 2, 3), plt.imshow(equalize, 'gray')
	plt.title('image'), plt.xticks([]), plt.yticks([])
	plt.subplot(2, 2, 4)
	plt.plot(hist2, color='k')
	plt.plot(cdf2, color='r')
	plt.title('histogram'), plt.xlim([0,256])
	plt.show()

	#cv2.imwrite('equalize.jpg', equalize)