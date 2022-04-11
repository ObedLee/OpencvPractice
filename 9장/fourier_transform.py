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
	nrows = cv2.getOptimalDFTSize(rows)
	ncols = cv2.getOptimalDFTSize(cols)

	right = ncols - cols
	bottom = nrows - rows
	bordertype = cv2.BORDER_CONSTANT
	img = cv2.copyMakeBorder(img, 0, bottom, 0, right, \
				bordertype, value=0)

	return img

def DFT_Numpy(img):
	img = convert_to_optimal_size(img)

	ft = np.fft.fft2(img)
	magnitude = np.abs(ft)
	ft_shift = np.fft.fftshift(ft)
	shift_magnitude = np.abs(ft_shift)

	return ft_shift, magnitude, shift_magnitude

def DFT_OpenCV(img):
	img = convert_to_optimal_size(img)

	ft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
	magnitude = cv2.magnitude(ft[:,:,0], ft[:,:,1])
	ft_shift = np.fft.fftshift(ft)
	shift_magnitude = cv2.magnitude(ft_shift[:,:,0], ft_shift[:,:,1])

	return ft_shift, magnitude, shift_magnitude

def log_magnitude(magnitude, k=1):
	log_mag = k*np.log(1 + magnitude)
	return log_mag

def IDFT_Numpy(data, sz):
	ishift = np.fft.ifftshift(data)
	img_back = np.fft.ifft2(ishift)
	img_back = np.real(img_back)
	img_back = img_back[0:sz[0], 0:sz[1]]
	return img_back

def IDFT_OpenCV(data, sz):
	ishift = np.fft.ifftshift(data)
	img_back = cv2.idft(ishift)
	img_back = img_back[0:sz[0], 0:sz[1], 0]
	return img_back

def make_filter(size, type='butterworth', cut_off=0.05, order=10):
	filt = np.zeros(size)
	if type=='low_ideal':
		filt = ideal(size, cut_off)
	elif type=='low_butterworth':
		filt = butterworth(size, cut_off, order)
	elif type=='low_Gaussian':
		filt = Gaussian(size, cut_off, order)
	elif type=='high_ideal':
		filt = ideal(size, cut_off)
		filt = 1 - filt
	elif type=='high_butterworth':
		filt = butterworth(size, cut_off, order)
		filt = 1 - filt
	elif type=='high_Gaussian':
		filt = Gaussian(size, cut_off, order)
		filt = 1 - filt

	return filt

def ideal(size, cut_off):
	crow, ccol = size[0]//2 , size[1]//2
	srow = np.int(size[0]*cut_off/2)
	scol = np.int(size[1]*cut_off/2)

	filt = np.zeros(size)
	filt[crow-srow:crow+srow+1, ccol-scol:ccol+scol+1] = 1

	return filt

def butterworth(size, cut_off, order):
	crow, ccol = size[0]//2 , size[1]//2
	std = np.uint8(max(size) // 2 * cut_off)

	(R, C) = np.meshgrid(np.linspace(0, size[0]-1, size[0]),
		np.linspace(0, size[1]-1, size[1]),
		sparse=False, indexing='ij')

	Duv = (((R-crow)**2+(C-ccol)**2)).astype(float)
	filt = 1/(1+(Duv/std**2)**order)

	return filt

def Gaussian(size, cut_off, order):
	crow, ccol = size[0]//2 , size[1]//2
	std = np.uint8(max(size) * cut_off)

	(R, C) = np.meshgrid(np.linspace(0, size[0]-1, size[0]),
		np.linspace(0, size[1]-1, size[1]),
		sparse=False, indexing='ij')

	Duv = (((R-crow)**2+(C-ccol)**2)).astype(float)
	filt = np.exp(-Duv/(2*std**2))

	return filt

def apply_filter(fshift, lp_filt):
	if(len(fshift.shape) is 2):
		n_lp_filt = lp_filt
	elif(len(fshift.shape) is 3):
		n_lp_filt = np.dstack([lp_filt, lp_filt])

	n_fshift = n_lp_filt * fshift

	return n_fshift

if __name__ == "__main__" :
	# conda activate psypy3
	# 명령행 인자 처리
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--image', required = True, \
			help = 'Path to the input image')
	args = vars(ap.parse_args())

	filename = args['image']

	# OpenCV를 사용하여 영상 데이터 로딩
	image = cv2.imread(filename)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# 변환 함수 유형 입력 처리
	print("변환 함수 종류 선택")
	print("  11. Numpy 제공 함수")
	print("  12. OpenCV 제공 함수")
	fun_type = eval(input("선택 >> "))

	# 필터 유형 입력 처리
	print("필터 종류 선택")
	print("  21. Lowpass (Ideal)")
	print("  22. Lowpass (Butterworth)")
	print("  23. Lowpass (Gaussian)")
	print("  24. Highpass (Ideal)")
	print("  25. Highpass (Butterworth)")
	print("  26. Highpass (Gaussian)")
	filter_type = eval(input("선택 >> "))

	if(fun_type < 11 or fun_type > 12 or \
		filter_type < 21 or filter_type > 26):
		print("Invalid input")
		exit(1)

	if(fun_type == 11):
		ft_shift, mag, s_mag = DFT_Numpy(gray)
	elif(fun_type == 12):
		ft_shift, mag, s_mag = DFT_OpenCV(gray)
	log_mag = log_magnitude(mag)
	log_s_mag = log_magnitude(s_mag)

	cutoff, order = 0.05, 5
	size = ft_shift.shape[0:2]
	if(filter_type == 21):
		filt = make_filter(size, 'low_ideal', cutoff)
	elif(filter_type == 22):
		filt = make_filter(size, 'low_butterworth', cutoff, order)
	elif(filter_type == 23):
		filt = make_filter(size, 'low_Gaussian', cutoff, order)
	elif(filter_type == 24):
		filt = make_filter(size, 'high_ideal', cutoff)
	elif(filter_type == 25):
		filt = make_filter(size, 'high_butterworth', cutoff, order)
	elif(filter_type == 26):
		filt = make_filter(size, 'high_Gaussian', cutoff, order)
	ft_filt_shift = apply_filter(ft_shift, filt)

	rows, cols = gray.shape
	if(fun_type == 11):
		filtered_log_s_mag = log_magnitude(np.abs(ft_filt_shift))
		image_rec = IDFT_Numpy(ft_filt_shift, (rows, cols))
	elif(fun_type == 12):
		magnitude = \
			cv2.magnitude(ft_filt_shift[:,:,0], ft_filt_shift[:,:,1])
		filtered_log_s_mag = log_magnitude(magnitude)
		image_rec = IDFT_OpenCV(ft_filt_shift, (rows, cols))

	plt.subplot(421),plt.imshow(gray, cmap = 'gray')
	plt.title('Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(422),plt.imshow(image_rec, cmap = 'gray')
	plt.title('Recon Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(423),plt.imshow(mag, cmap = 'gray')
	plt.title('Spectrum'), plt.xticks([]), plt.yticks([])
	plt.subplot(424),plt.imshow(log_mag, cmap = 'gray')
	plt.title('Log Spectrum'), plt.xticks([]), plt.yticks([])
	plt.subplot(425),plt.imshow(s_mag, cmap = 'gray')
	plt.title('Shifted Spectrum'), plt.xticks([]), plt.yticks([])
	plt.subplot(426),plt.imshow(log_s_mag, cmap = 'gray')
	plt.title('Log Shifted Spectrum'), plt.xticks([]), plt.yticks([])
	plt.subplot(428),plt.imshow(filtered_log_s_mag, cmap = 'gray')
	plt.title('filtered LS Spectrum'), plt.xticks([]), plt.yticks([])
	plt.show()