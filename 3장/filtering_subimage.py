# 필요한 패키지를 import함
from __future__ import print_function
import argparse
import cv2
import numpy as np
from handle_channel_roi import display_image

def filtering_ROI(img, rect):
	# 입력 영상을 복사하여 결과 영상 생성
	dst = img.copy()

	if len(img.shape) == 2 :
		type = 1
	elif len(img.shape) == 3 and \
		img.shape[2] == 3:
		type = 2

	roi = dst[rect[1]:rect[3], rect[0]:rect[2]]
	if type == 2:
		roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	roi_canny = cv2.Canny(roi_gray, 100, 200)

	if type == 2:
		roi_canny = cv2.cvtColor(roi_canny, cv2.COLOR_GRAY2BGR)
	dst[rect[1]:rect[3], rect[0]:rect[2]] = roi_canny

	return dst

if __name__ == '__main__' :
	# 명령행 인자 처리
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required = True, \
			help = "Path to the input image")
	ap.add_argument("-s", "--start_point", type = int, \
 			nargs='+', default=[0, 0], \
			help = "Start point of the rectangle")
	ap.add_argument("-e", "--end_point", type = int, \
	 		nargs='+', default=[50, 50], \
			help = "End point of the rectangle")
	args = vars(ap.parse_args())

	filename = args["image"]
	sp = args["start_point"]
	ep = args["end_point"]

	# OpenCV를 사용하여 영상 데이터 로딩
	image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

	rect = sp + ep
	filtered = filtering_ROI(image, rect)

	display_image(filtered, 'filtering')





