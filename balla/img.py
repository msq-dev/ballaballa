import cv2 as cv
import numpy as np


class BallaImage:
    def __init__(self, img, name: str = "Ballaballa"):
        self.img = img
        self.name = name

    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, img):
        self._img = img

    def blur(self, blur_radius: int = 3):
        if len(self.img.shape) != 2:
            self.img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        kernel_size = (blur_radius, blur_radius)
        self.img = cv.GaussianBlur(self.img, kernel_size, 1)

    def draw(self, contours):
        self.img = cv.cvtColor(self.img, cv.COLOR_GRAY2BGR)
        cv.drawContours(self.img, contours, -1, (0, 255, 0), 3)

    def warp(self, corner_points,
             result_width: int = 500,
             result_height: int = 500):
        """Adjust perspective"""

        width = result_width
        height = result_height
        points = corner_points.reshape((4, 2))
        points_new = np.zeros((4, 1, 2), np.int32)
        add = points.sum(1)

        # starting point -> smallest value in "add"
        start_point = min(add)
        start_point_index = np.where(add == start_point)[0].item(0)

        diff = np.diff(points, axis=1)

        points_new[0] = points[start_point_index]   # [0, 0]
        points_new[1] = points[np.argmin(diff)]     # [width, 0]
        points_new[2] = points[np.argmax(diff)]     # [0, height]
        points_new[3] = points[np.argmax(add)]      # [width, height]

        points_01 = np.float32(points_new)
        points_02 = np.float32(
            [
                [0, 0],
                [width, 0],
                [0, height],
                [width, height]
            ]
        )
        matrix = cv.getPerspectiveTransform(points_01, points_02)
        self.img = cv.warpPerspective(self.img, matrix, (width, height))

    def simple_threshold(self, threshold):
        _, self.img = cv.threshold(self.img, threshold, 255, cv.THRESH_BINARY)

    def show(self):
        cv.imshow(self.name, self.img)
        cv.waitKey(0)
