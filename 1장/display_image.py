# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:37:06 2022

@author: dmz01
"""
from __future__ import print_function
import argparse
import cv2


def display_image(img):
    # 영상 정보 출력
    print("(rows, cols, ch): {}".format(img.shape))

    # 영상 출력을 위한 윈도우 생성
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    # 윈도우에 영상 출력
    cv2.imshow('image', img)

    # 사용자 입력 대기
    cv2.waitKey(0)

    # 윈도우 파괴
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 명령행 인자 처리
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="Path to the input image")
    args = vars(ap.parse_args())

    filename = args["image"]

    # opencv를 사용하여 영상 데이터 로딩
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    if image is None:
        print('{}: reading error'.format(filename))
    else:
        # 윈도우에 영상 출력
        display_image(image)
