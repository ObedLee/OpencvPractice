# 필요한 패키지를 import함
from __future__ import print_function
import argparse
import cv2
import numpy as np
import random


def mosaic_image(img, rect, size=(5, 5), mtype=1):
	# 함수 정의
	dst = img.copy()

	if mtype == 1:
		for row in range(rect[0], rect[2], size[0]):
			for col in range(rect[1], rect[3], size[1]):
				arr = np.array(dst[row:row+size[0], col:col+size[1]])
				arr = arr[1:3, 0:2]
				arr = np.array(np.mean(arr, 1))
				arr = np.mean(arr, 0)
				dst[row:row+size[0], col:col+size[1]] = arr
	elif mtype == 2:
		for row in range(rect[0], rect[2], size[0]):
			for col in range(rect[1], rect[3], size[1]):
				arr = np.array(dst[row:row+size[0], col:col+size[1]])
				arr = arr[1:3, 0:2]
				arr = np.array(np.max(arr, 1))
				arr = np.max(arr, 0)
				dst[row:row + size[0], col:col + size[1]] = arr
	elif mtype == 3:
		for row in range(rect[0], rect[2], size[0]):
			for col in range(rect[1], rect[3], size[1]):
				arr = np.array(dst[row:row+size[0], col:col+size[1]])
				arr = arr[1:3, 0:2]
				arr = np.array(np.min(arr, 1))
				arr = np.min(arr, 0)
				dst[row:row + size[0], col:col + size[1]] = arr
	elif mtype == 4:
		for row in range(rect[0], rect[2], size[0]):
			for col in range(rect[1], rect[3], size[1]):
				rand_row = random.randrange(row, row + size[0])
				rand_col = random.randrange(col, col + size[1])
				dst[row:row + size[0], col:col + size[1]] = dst[rand_row, rand_col]

	return dst


if __name__ == '__main__' :
	# 명령행 인자 처리
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required = True,
					help = "Path to the input image")
	ap.add_argument("-s", "--start_point", type = int,
					nargs='+', default=[0, 0],
					help = "Start point of the rectangle")
	ap.add_argument("-e", "--end_point", type = int,
					nargs='+', default=[150, 100],
					help = "End point of the rectangle")
	ap.add_argument("-z", "--size", type = int,
					nargs='+', default=[15, 15],
					help = "Mosaic Size")
	ap.add_argument("-t", "--type", type = int,
					default=1,
					help = "Mosaic Type")
	args = vars(ap.parse_args())

	filename = args["image"]
	sp = args["start_point"]
	ep = args["end_point"]
	size = args["size"]
	mtype = args["type"]

	# OpenCV를 사용하여 영상 데이터 로딩
	image = cv2.imread(filename, cv2.IMREAD_COLOR)
	if image is None :
		raise IOError("Cannot open the image")

	(rows, cols, _) = image.shape
	if sp[0] < 0 or sp[1] < 0 or ep[0] > rows or ep[1] > cols :
		raise ValueError('Invalid Size')

	# list 연결
	rect = sp + ep

	# 모자이크 영상 생성
	result = mosaic_image(image, rect, size, mtype)

	# 영상 출력을 윈도우 생성
	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	# 윈도우에 영상 출력
	cv2.imshow('image', result)

	# 사용자 입력 대기
	cv2.waitKey(0)
	# 윈도우 파괴
	cv2.destroyAllWindows()