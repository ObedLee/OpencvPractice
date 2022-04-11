import argparse
import cv2
from matplotlib import pyplot as plt

if __name__ == "__main__" :
	# 명령행 인자 처리
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--image', required = True, \
			help = 'Path to the input image')
	args = vars(ap.parse_args())

	filename = args['image']

	# OpenCV를 사용하여 영상 데이터 로딩
	image = cv2.imread(filename)

	# 그레이스케일 영상으로 변환하고 노이즈 제거
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.medianBlur(image, 5)

	th, dst1 = cv2.threshold(image, 127, 255, \
	 cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	dst2 = cv2.adaptiveThreshold(image, 255, \
		cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
	dst3 = cv2.adaptiveThreshold(image, 255, \
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

	titles = ['image', 'Global Thresh', 'Adaptive Mean Thresh', \
	 'Adaptive Gaussian Thresh']
	images = [image, dst1, dst2, dst3]

	for i in range(4):
		plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
		plt.title(titles[i])
		plt.xticks([]),plt.yticks([])

	plt.show()