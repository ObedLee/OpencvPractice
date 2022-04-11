# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 10:04:57 2022

@author: dmz01
"""

import cv2


def save_image():
    image = cv2.imread("../images/nature.jpg")
    cv2.imshow('window', image)
    pk = cv2.waitKey(0) & 0xFF  # in 64bit machine

    if pk == ord('s'):  # chr(99)는 문자  'c' 반환
        cv2.imwrite('copy.jpg', image)


if __name__ == "__main__":
    save_image()
