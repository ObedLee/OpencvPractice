# 필요한 패키지를 import함
from __future__ import print_function
import argparse
import PPM.PPM_P6 as ppm

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True, \
		help = "Path to the input image")
ap.add_argument("-o", "--output", required = True, \
		help = "Path to the output image")
args = vars(ap.parse_args())

infile = args["input"]
outfile = args["output"]

# PPM_P6 객체 생성
ppm_p6 = ppm.PPM_P6()

# PPM_P6 객체를 사용하여 PPM 파일 읽기
(width, height, maxval, bitmap) = ppm_p6.read(infile)
# PPM_P6 객체 정보 출력
print(ppm_p6)

# PPM_P6 객체를 사용하여 PPM 파일 저장
ppm_p6.write(width, height, maxval, bitmap, outfile)