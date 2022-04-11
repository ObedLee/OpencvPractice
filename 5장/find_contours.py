# 필요한 패키지를 import함
from __future__ import print_function
from random import seed, randint
import argparse
import cv2
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

def findContours(img):
	# 타원형의 구조적 요소 생성
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

	# 열림 연산과 닫힘 연산을 순차적으로 적용
	img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
	img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

	# contour 생성
	version = int(cv2.__version__.split(".")[0])
	if version == 2 or version == 4:
		(contours, hierarchy) = cv2.findContours(img, \
					cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	elif version == 3:
		(_, contours, hierarchy) = cv2.findContours(img, \
					cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	print("Total number of contours: ", len(contours))

	# 무작위 색으로 모든 연결요소의 외곽선 그림
	seed(9001)
	contour_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	for (i, c) in enumerate(contours):
		r = randint(0, 256)
		g = randint(0, 256)
		b = randint(0, 256)
		cv2.drawContours(contour_img, [c], 0, (b,g,r), 2)

	return contour_img, contours

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
	contour_img, contours = findContours(binary)

	# 0번(1st) 연결요소의 외곽선만을 포함하는 영상 생성
	new_img = np.zeros_like(image, dtype="uint8")
	cntr = sorted(contours, key = cv2.contourArea, reverse = True)[0]
	cv2.drawContours(new_img, [cntr], 0, (255,0,0), 3)
	print("면적: ", cv2.contourArea(cntr))

	# 복제 영상 생성
	box_img = deepcopy(new_img)

	# Convex Hull 표시
	hull = cv2.convexHull(cntr)
	cv2.drawContours(new_img, [hull], 0, (0,255,0), 1)

	# 최소 사각형(minimum bounding rectangle) 표시
	# 1. minimum bounding rectangle
	(x,y,w,h) = cv2.boundingRect(cntr)
	cv2.rectangle(box_img, (x,y), (x+w,y+h), (255,255,0), 1) 
	# 2. minimum and rotated bounding rectangle
	# (c_x, c_y), (width, height), angle
	rect = cv2.minAreaRect(cntr)
	box = np.int0(cv2.boxPoints(rect))
	cv2.drawContours(box_img, [box], 0, (0,255,255), 1)
	loc = tuple(box[3])
	cv2.putText(box_img, "angle: {}".format(np.int0(rect[2])), \
		loc, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

	# 결과 영상 출력
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	plt.subplot(2, 2, 1), plt.imshow(image)
	plt.title('image'), plt.xticks([]), plt.yticks([])
	plt.subplot(2, 2, 2), plt.imshow(contour_img)
	plt.title('labeled image'), plt.xticks([]), plt.yticks([])
	plt.subplot(2, 2, 3), plt.imshow(new_img)
	plt.title('largest contour'), plt.xticks([]), plt.yticks([])
	plt.subplot(2, 2, 4), plt.imshow(box_img)
	plt.title('bounding shape'), plt.xticks([]), plt.yticks([])
	plt.show()