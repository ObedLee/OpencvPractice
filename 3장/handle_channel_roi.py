# 필요한 패키지를 import함
from __future__ import print_function
import argparse
import cv2

def display_image(img, winname, flag=True):
	# 영상 정보 출력
	print("(rows, cols, ch): {}".format(img.shape))

	# 영상 출력을 윈도우 생성
	cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
	# 윈도우에 영상 출력
	cv2.imshow(winname, img)

	# 사용자 입력 대기
	if flag:
		cv2.waitKey(0)

def handle_ROI(img, rect):
	# 입력 범위의 유효성 확인
	[sx, sy, ex, ey] = rect
	if(sx < 0 or sy < 0 or
		ex >= img.shape[1] or
		ey >= img.shape[0]):
		print("[에러]유효하지 않은 범위")
		return None

	# 영상에서 부분 영상 추출
	# Numpy의 슬라이싱 기법 사용
	crop = img[sy:ey, sx:ex]

	return crop

def split_channel(img):
	b, g, r = cv2.split(img)

	# Numpy의 슬라이싱 기법 사용
	# b = img[:, :, 0]
	# g = img[:, :, 1]
	# r = img[:, :, 2]

	return r, g, b

def merge_channel(r, g, b):
	img = cv2.merge((b, g, r))

	# Numpy의 슬라이싱 기법 사용
	# img[:, :, 0] = b
	# img[:, :, 1] = g
	# img[:, :, 2] = r

	return img


if __name__ == '__main__' :
	# 명령행 인자 처리
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required = True, \
			help = "Path to the input image")
	args = vars(ap.parse_args())

	filename = args["image"]

	# OpenCV를 사용하여 영상 데이터 로딩
	image = cv2.imread(filename, cv2.IMREAD_COLOR)

	# 부분 영상(ROI) 추출
	subimage = handle_ROI(image, [50, 50, 200, 100])
	if(subimage is None):
		print("[에러]프로그램을 종료합니다.")
		exit(1)

	# 채널 분리
	r, g, b = split_channel(subimage)

	# 채널별 영상을 윈도우에 출력
	display_image(r, "R", False)
	display_image(g, "G", False)
	display_image(b, "B", False)

	# 채널별 영상 병합
	recon = merge_channel(r, g, b)

	# 채널 병합한 영상을 윈도우에 출력
	display_image(recon, "recon")