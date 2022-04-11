# 필요한 패키지를 import함
import argparse
import array
import numpy as np
import PPM.PPM_P6 as ppm

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="Path to the input image")
ap.add_argument("-l", "--location", type=int,
                nargs='+', default=[0, 0],
                help="Location of the rect in the output image")
ap.add_argument("-s", "--size", type=int,
                nargs='+', default=[50, 50],
                help="Size of the output image")
ap.add_argument("-c", "--color", type=int,
                nargs='+', default=[255, 0, 0],
                help="Color of each pixel in the output image")
ap.add_argument("-o", "--output", required=True,
                help="Path to the output image")
args = vars(ap.parse_args())

infile = args["input"]
outfile = args["output"]
location = args["location"]
size = args["size"]
color = args["color"]

y_start = location[1]
y_end = location[1] + size[1]
x_start = location[0]
x_end = location[0] + size[0]

# PPM_P6 객체 생성
ppm_p6 = ppm.PPM_P6()

# PPM_P6 객체를 사용하여 PPM 파일 읽기
(width, height, maxval, bitmap) = ppm_p6.read(infile)

# PPM_P6 객체 정보 출력
print(ppm_p6)

# bytres 데이터 이미지를 3차원 numpy array형으로 저장
image = array.array('B', bitmap)
image = np.array(image)
image = image.reshape((height, width, 3))

# 이미지에 color rect 그리기
image[y_start:y_end, x_start:x_end] = color

# 3차원 array를 1차원 array로  변환 후 bytes 데이터로 변환
image = image.reshape(height * width * 3)
image = bytes(image)

# PPM_P6 객체를 사용하여 PPM 파일 저장
ppm_p6.write(width, height, maxval, image, outfile)
