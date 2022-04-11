# import the necessary packages
from __future__ import print_function
import numpy as np
import argparse
import glob
import cv2
import os

def detectBarcode(img, verbose=False):

	# 코드 작성

	return points

if __name__ == "__main__" :
	# 명령행 인자 처리
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", required = True, \
		help = "Path to dataset folder")
	args = vars(ap.parse_args())

	dataset = args["dataset"]

	if(not os.path.isdir("results")):
		os.mkdir('results')

	verbose = True

	# dataset 폴더에서 jpg 파일만을 검출
	for imagePath in glob.glob(dataset + "/*.jpg"):
		print(imagePath, ' 처리중...')

		# 영상을 불러오고 그레이스케일 영상으로 변환
		image = cv2.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# 바코드 검출
		points = detectBarcode(gray, verbose)

		# 바코드 영역 표시
		cv2.drawContours(image, [points], -1, (0, 255, 0), 3)

		# 결과 영상 저장
		loc1 = imagePath.rfind("\\")
		loc2 = imagePath.rfind(".")
		fname = 'results/' + imagePath[loc1+1:loc2] + '_res.jpg'
		cv2.imwrite(fname, image)

		if verbose:
			cv2.imshow("Image", image)
			cv2.waitKey(0)