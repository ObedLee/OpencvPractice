# import the necessary packages
import argparse
import pickle
import glob
import cv2

def histogram1D(img):
	# 1차원 히스토그램 계산
	hist = cv2.calcHist([img], [0], None, [256], [0, 256])
	hist = cv2.normalize(hist, hist)

	# 결과 히스토그램 반환
	return hist

def histogram2D(img):
	# 2차원 히스토그램 계산
	hist = cv2.calcHist( [img], [0, 1], None, \
				[180, 256], [0, 180, 0, 256] )
	hist = cv2.normalize(hist, hist)

	# 결과 히스토그램 반환
	return hist

def histogram3D(img):
	# 2차원 히스토그램 계산
	hist = cv2.calcHist( [img], [0, 1, 2], None, \
				[32, 32, 32], [0, 256, 0, 256, 0, 256] )
	hist = cv2.normalize(hist, hist)

	# 결과 히스토그램 반환
	return hist

if __name__ == "__main__" :
	# 명령행 인자 처리
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", required = True,
		help = "Path to the directory that contains the images to be indexed")
	ap.add_argument("-i", "--index", required = True,
		help = "Path to where the computed index will be stored")
	args = vars(ap.parse_args())

	# dictionary를 사용하여 추출한 영상별 특징 정보 저장
	# 키: 영상 파일명, 값: 히스토그램 기반의 특징 정보
	index = {}

	# glob를 사용하여 영상 경로 추출
	for imagePath in glob.glob(args["dataset"] + "/*.jpg"):
		# 영상 파일명 추출
		fname = imagePath[imagePath.rfind("/") + 1:]

		# 히스토그램을 계산하여 dictionary에 저장
		image = cv2.imread(imagePath)
		hist = histogram2D(image)
		index[fname] = hist

	# 특징 정보를 파일에 저장
	f = open(args["index"], "wb")
	pickle.dump(index, f)
	f.close()

	# 총 생성된 파일 개수 출력
	print ("done...indexed %d images" % (len(index)))