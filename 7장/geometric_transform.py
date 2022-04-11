# 필요한 패키지를 import함
from __future__ import print_function
import argparse
import cv2
import numpy as np
import math

def Affine_transform(img, type):
	# 입력 영상의 크기 저장
	rows, cols, _ = img.shape

	# 결과 영상 크기 설정을 위한 차분값
	dx = dy = 0

	if type == 1:
		dx = 100
		dy = 50
		matrix = np.float32([[1, 0, dx], [0, 1, dy]])
	elif type == 2:
		matrix = cv2.getRotationMatrix2D((cols/2,rows/2), 45, 1)
	elif type == 3:
		matrix = np.float32([[1.5, 0, 0], [0, 1.5, 0]])
	elif type == 4:
		u = math.tan(30*math.pi/180)
		matrix = np.float32([[1, u, 0], [0, 1, 0]])

		# 변환 후의 영상 크기 변경 정도 계산
		newpt = np.atleast_2d([cols-1, rows-1, 0])
		newpt = newpt.transpose()
		newpt = matrix.dot(newpt)
		dx = newpt[0] - cols
	elif type == 5:
		pts1 = np.float32([[50,50], [200,50], [50,200]])
		pts2 = np.float32([[10,100], [200,50], [100,250]])
		matrix = cv2.getAffineTransform(pts1, pts2)

	dst = cv2.warpAffine(img, matrix, (cols+dx, rows+dy), flags=cv2.WARP_INVERSE_MAP)

	return dst

def perspective_transform(img):
	# 입력 영상의 크기 저장
	rows, cols, _ = img.shape

	pts1 = np.float32([[107, 107], [414, 68], [130, 386], [410, 406]])
	pts2 = np.float32([[100, 100], [400, 100], [100, 400], [400, 400]])
	matrix = cv2.getPerspectiveTransform(pts1, pts2)
	dst = cv2.warpPerspective(img, matrix, (cols, rows))

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

	# 변환 유형 입력 처리
	print("공간변환 종류 선택")
	print("  1. 평행이동 (x: 100, y: 50)")
	print("  2. 회전 (반시계방향 45도)")
	print("  3. 스케일링 (1.5배 확대)")
	print("  4. 비틀기 (x축 30도)")
	print("  5. 대응점에 의한 변환")
	print("  6. 투영 변환")
	type = eval(input("선택 >> "))

	if type > 6:
		print("Invalid input")
		exit(1)
	elif type == 6:
		dst = perspective_transform(image)
	elif (type >= 1 and type <= 5):
		dst = Affine_transform(image, type)

	# 결과 영상 출력
	cv2.imshow("images", image)
	cv2.imshow("transformed", dst)
	cv2.waitKey(0)