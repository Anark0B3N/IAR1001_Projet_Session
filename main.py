# coding=utf-8
# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import numpy as np

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    img = cv2.imread('letters.jpg', -1)
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) [-2:]
    cv2.drawContours(img, contours, -1, (0,255,0), 3)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('image_contours.jpg', img)
    print("Hi, {0}".format(name))  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
