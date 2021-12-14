import cv2 as cv
import numpy as np


def area(contour) -> float:
    return cv.contourArea(contour)


def show(img, window_title: str = "") -> None:
    cv.imshow(window_title, img)
    cv.waitKey(0)


def find_contours(img):
    cont, hierarchy = cv.findContours(
        img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return cont


colors = {
    "MAGENTA": (255, 0, 255),
    "GREEN": (0, 255, 0),
    "BLUE": (255, 0, 0),
    "RED": (0, 0, 255),
    "CYAN": (255, 255, 0)
}
