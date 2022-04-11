# 필요한 패키지를 import함
from __future__ import print_function
import argparse
import cv2

def get_pixel(img, x, y):
	# 좌표 유효성 확인
	if (x >= img.shape[1] or
		y >= image.shape[0]):
		print("Invalide coord. ({}, {})".format(x, y))
		return None

	# 픽셀 값 반환
	return img[y, x]

def put_pixel(img, x, y, value):
	# 좌표 유효성 확인
	if (x >= img.shape[1] or
		y >= image.shape[0]):
		print("Invalide coord. ({}, {})".format(x, y))

	img[y, x] = value;

if __name__ == '__main__' :
	# 명령행 인자 처리
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required = True, \
			help = "Path to the input image")
	args = vars(ap.parse_args())

	filename = args["image"]

	# OpenCV를 사용하여 영상 데이터 로딩
	image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

	# 픽셀 값 접근
	value = get_pixel(image, 30, 20)
	print("[Before]value: {}".format(value))

	put_pixel(image, 30, 20, 0)
	value = get_pixel(image, 30, 20)
	print("[After]value: {}".format(value))