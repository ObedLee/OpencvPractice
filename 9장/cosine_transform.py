# 필요한 패키지를 import함
from __future__ import print_function
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt

def convert_to_optimal_size(img):
	# Convert to optimal size by padding 0s
	# It is fastest when array size is power of two
	# size is a product of 2’s, 3’s, and 5’s are also good
	rows, cols = img.shape
	nrows = 2*cv2.getOptimalDFTSize((rows+1)//2)
	ncols = 2*cv2.getOptimalDFTSize((cols+1)//2)

	right = ncols - cols
	bottom = nrows - rows
	bordertype = cv2.BORDER_CONSTANT
	img = cv2.copyMakeBorder(img, 0, bottom, 0, right, \
				bordertype, value=0)

	return img

def DCT_OpenCV(img):
	img = convert_to_optimal_size(img)

	dct = cv2.dct(np.float32(img))
	magnitude = np.abs(dct)

	return dct, magnitude

def IDCT_OpenCV(data, sz):
	img_back = cv2.idct(data)
	img_back = np.uint8(img_back[0:sz[0], 0:sz[1]])
	return img_back

def log_magnitude(magnitude, k=1):
	log_mag = k*np.log(1 + magnitude)
	return log_mag

if __name__ == "__main__" :
	# 명령행 인자 처리
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--image', required = True, \
			help = 'Path to the input image')
	args = vars(ap.parse_args())

	filename = args['image']

	# OpenCV를 사용하여 영상 데이터 로딩
	image = cv2.imread(filename)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	dct, mag = DCT_OpenCV(gray)
	log_mag = log_magnitude(mag)

	rows, cols = gray.shape
	image_rec = IDCT_OpenCV(dct, (rows, cols))

	plt.subplot(221),plt.imshow(gray, cmap = 'gray')
	plt.title('Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(222),plt.imshow(image_rec, cmap = 'gray')
	plt.title('Recon Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(223),plt.imshow(mag, cmap = 'gray')
	plt.title('Spectrum'), plt.xticks([]), plt.yticks([])
	plt.subplot(224),plt.imshow(log_mag, cmap = 'gray')
	plt.title('Log Spectrum'), plt.xticks([]), plt.yticks([])
	plt.show()