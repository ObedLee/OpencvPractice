# 필요한 패키지를 import함
from __future__ import print_function
import argparse
import cv2
from matplotlib import pyplot as plt

def histogram(img):
	# 히스토그램 계산
	histg = cv2.calcHist([img], [0], None, [256], [0, 256])

	# 결과 히스토그램 반환
	return histg

if __name__ == '__main__' :
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

	# 히스토그램 계산
	hist = histogram(gray)

	# 히스토그램 출력
	plt.subplot(1, 2, 1), plt.imshow(gray, 'gray')
	plt.title('image')

	plt.subplot(1, 2, 2)
	plt.plot(hist, color='blue', marker='^', linestyle='solid')
	plt.title('histogram'), plt.xlim([0,256])
	plt.grid(True, color='0.5', linestyle='dashed', linewidth=0.5)

	plt.show()