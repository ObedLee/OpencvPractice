# 필요한 패키지를 import함
from __future__ import print_function
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt

def findConnectedComponents(img, conn=8):
	# 타원형의 구조적 요소 생성
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

	# 2번 반복하여 침식 연산과 팽창 연산을 순차적으로 적용
	img = cv2.erode(img, kernel, iterations = 2)
	img = cv2.dilate(img, kernel, iterations = 2)

	# 연결요소 생성
	# 0번 레이블은 배경에 해당하고 객체들은 1번 레이블부터 할당
	num, labels = cv2.connectedComponents(img, connectivity = conn)
	print("Total number of components: ", num-1)

	# 연결요소의 라벨을 HSV 칼라공간의 색상(H)으로 변환
	hue = np.uint8(179*labels/np.max(labels))

	# hue 배열과 동일한 크기를 가지며 255로 채워진 배열 생성
	new_ch = 255*np.ones_like(hue)

	# HSV 칼라 영상 생성 (채도와 명도는 최대값으로 설정)
	labeled_img = cv2.merge([hue, new_ch, new_ch])

	# 화면 표시를 위해 HSV 칼라공간을 BGR 칼라공간의 변환
	labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2RGB)

	# 배경 라벨(0)에 해당하는 픽셀은 검은색으로 설정
	labeled_img[hue==0] = [0, 0, 0]

	return labeled_img, labels

if __name__ == "__main__" :
	# 명령행 인자 처리
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--image', required = True, \
			help = 'Path to the input image')
	args = vars(ap.parse_args())

	filename = args['image']

	# OpenCV를 사용하여 영상 데이터 로딩
	image = cv2.imread(filename)

	# Grayscale 영상으로 변환한 후
	# 가우시안 평활화 및 임계화 수행
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	th, binary = cv2.threshold(gray, 0, 255, \
		 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	# 연결요소 생성
	labeled_img, labels = findConnectedComponents(binary)

	# 1번 라벨의 연결요소만을 포함하는 영상 생성
	new_img = np.zeros_like(labels, dtype="uint8")
	new_img[labels==1] = 255

	# 결과 영상 출력
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	plt.subplot(1, 3, 1), plt.imshow(image)
	plt.title('image'), plt.xticks([]), plt.yticks([])
	plt.subplot(1, 3, 2), plt.imshow(labeled_img)
	plt.title('labeled image'), plt.xticks([]), plt.yticks([])
	plt.subplot(1, 3, 3), plt.imshow(new_img, 'gray')
	plt.title('1st label image'), plt.xticks([]), plt.yticks([])
	plt.show()