import cv2 as cv
import numpy as np
from balla.img import BallaImage
from utils import area as A
from utils import show, find_contours, colors


class NoTargetError(Exception):
    def __init__(self, img) -> None:
        self.img = img
        canny = cv.Canny(self.img, 125, 255)
        contours, hierarchy = cv.findContours(
            canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        clr_img = cv.cvtColor(self.img, cv.COLOR_GRAY2BGR)
        cv.drawContours(clr_img, contours, -1, colors["GREEN"], 2)

        print("No target detected")
        show(clr_img)


class NoHitsError(Exception):
    def __init__(self, img) -> None:
        self.img = img

        print("No hits detected")
        show(self.img)


def detect_target(blurred):
    """Try to find largest rectangle (i.e. the target) in provided image by decrementing the Canny threshold, then return its corner points
    """

    blurred_img = blurred.img
    canny_threshold = 255
    rect_found = False

    while not rect_found and canny_threshold != 0:
        canny = cv.Canny(blurred_img, canny_threshold, 255)
        contours, _ = cv.findContours(canny,
                                      cv.RETR_EXTERNAL,
                                      cv.CHAIN_APPROX_SIMPLE)
        areas = [A(c) for c in contours if A(c) > 1000]

        if len(areas):
            for c in contours:
                if A(c) == sorted(areas)[-1]:  # largest area
                    perimeter = cv.arcLength(c, True)
                    corner_points = \
                        cv.approxPolyDP(c, 0.02 * perimeter, True)
                    if len(corner_points) == 4:
                        rect_found = True
                        return corner_points
        canny_threshold -= 1

    raise NoTargetError(blurred_img)


def detect_hits(warped):
    """Try to find circles with radius equal to bullet radius in provided image by decrementing the threshold, draw them on image and return their quantity and image
    """

    warped.blur()
    thresh = BallaImage(warped.img, "Thresh")
    threshold = 252
    all_hits_detected = False
    bullet_radius = list(range(2, 6))
    hits = []

    while not all_hits_detected and threshold != 200:
        thresh.simple_threshold(threshold)
        contours, _ = cv.findContours(thresh.img, cv.RETR_EXTERNAL,
                                      cv.CHAIN_APPROX_SIMPLE)

        if len(contours):
            if len(contours) == len(hits):
                all_hits_detected = True
                print(f"Done at threshold {threshold}")

            cont_polygon = [None]*len(contours)
            center = [None]*len(contours)
            radius = [None]*len(contours)

            for i, c in enumerate(contours):
                cont_polygon[i] = cv.approxPolyDP(c, 3, True)
                center[i], radius[i] = cv.minEnclosingCircle(cont_polygon[i])
                if int(radius[i]) in bullet_radius:
                    hits.append(c)
        threshold -= 1

    if not len(hits):
        raise NoHitsError(thresh)

    # hitpoints = 0
    # for index, hit in enumerate(hits):
    #     # calculate average center of hit
    #     # https://en.m.wikipedia.org/wiki/Image_moment
    #     mom = cv.moments(hit)
    #     center_X = int(mom["m10"] / mom["m00"])
    #     center_Y = int(mom["m01"] / mom["m00"])

    #     radius = int(cv.arcLength(hit, True) / 4)
    #     cv.circle(warped_img, (center_X, center_Y),
    #               1, colors["BLUE"], 2)
    #     cv.circle(warped_img, (center_X, center_Y),
    #               radius, colors["MAGENTA"], 2)
    # return len(hits), warped_img
