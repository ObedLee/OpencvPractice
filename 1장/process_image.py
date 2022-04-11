# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 15:06:18 2022

@author: dmz01
"""

from __future__ import print_function
import argparse
import cv2
import display_image


def process_image(image):
    result = cv2.GaussianBlur(image, (5, 5), 0)
    result = cv2.Canny(result, 100, 200)

    return result


if __name__ == '__main__':
    # 명령행 인자 처리
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
                    help="Path to the imput image")
    ap.add_argument("-o", "--output", required=True,
                    help="Path to the output image")
    args = vars(ap.parse_args())

    infile = args["input"]
    outfile = args["output"]

    # opencv를 사용하여 영상 데이터 로딩
    image = cv2.imread(infile, cv2.IMREAD_COLOR)
    if image is None:
        print('{}: reading error'.format(infile))
    else:
        # 영상 블러링 및 캐니 에지 연산 수행
        result = process_image(image)

        # 윈도우에 영상 출력
        display_image.display_image(result)

        # 결과 영상을 새로운 이름으로 저장
        cv2.imwrite(outfile, result)
