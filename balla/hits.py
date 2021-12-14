import cv2 as cv
from target import detect_sheet, warp_sheet
from utils import find_ring, draw_ring

MAGENTA = (255, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
CYAN = (255, 255, 0)


def detect_hits(img, bg, size):
    BG_MODE = {
        0: (230, cv.THRESH_BINARY, "white"),
        1: (20, cv.THRESH_BINARY_INV, "black")
    }

    sheet = detect_sheet(img)

    if sheet is None:
        return 0, img

    target = warp_sheet(img, sheet, size)

    gray = cv.cvtColor(target, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3, 3), 1)

    ###### RINGS ################
    ring, ring_radius = find_ring(blur)

    if ring.any() == 0:
        print("No ring detected.")
        return 0, target

    ring_mom = cv.moments(ring)
    ring_center_X = int(ring_mom["m10"] / ring_mom["m00"])
    ring_center_Y = int(ring_mom["m01"] / ring_mom["m00"])

    ring_center = (ring_center_X, ring_center_Y)

    resize_factor = 0.6055

    rings_radii = [
        int(
            round(
                ring_radius * (resize_factor - resize_factor / 10 * r)
            )
        ) for r in range(10)
    ]

    rings_radii.reverse()

    _10 = int(round(ring_radius * 0.06))
    # draw_ring(target, ring_center, _10)

    _9 = int(round(ring_radius * 0.12))
    # draw_ring(target, ring_center, _9)

    _8 = int(round(ring_radius * 0.18))
    # draw_ring(target, ring_center, _8)

    _7 = int(round(ring_radius * 0.25))
    # draw_ring(target, ring_center, _7)

    _6 = int(round(ring_radius * 0.31))
    # draw_ring(target, ring_center, _6)

    _5 = int(round(ring_radius * 0.37))
    # draw_ring(target, ring_center, _5)

    _4 = int(round(ring_radius * 0.43))
    # draw_ring(target, ring_center, _4)

    _3 = int(round(ring_radius * 0.49))
    # draw_ring(target, ring_center, _3)

    _2 = int(round(ring_radius * 0.545))
    # draw_ring(target, ring_center, _2)

    _1 = int(round(ring_radius * 0.6))
    # draw_ring(target, ring_center, _1)

    ###########################

    rings_radii_01 = [_10, _9, _8, _7, _6, _5, _4, _3, _2, _1]
    print(f"calculated: {rings_radii}")
    print(f"madebyhand: {rings_radii_01}")

    retval, thresh = cv.threshold(
        blur, BG_MODE[bg][0], 255, BG_MODE[bg][1])

    hits, hierarchy = cv.findContours(
        thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    if not len(hits):
        print("No hits detected.")
        print(
            f"Is {bg}(='{BG_MODE[bg][2]}') the appropriate background color?")
        return 0, target

    hitpoints = 0
    for index, h in enumerate(hits):
        # calculate average center of hit
        # https://en.m.wikipedia.org/wiki/Image_moment
        mom = cv.moments(h)
        center_X = int(mom["m10"] / mom["m00"])
        center_Y = int(mom["m01"] / mom["m00"])

        radius = int(cv.arcLength(h, True) / 4)
        cv.circle(target, (center_X, center_Y),
                  1, BLUE, 2)
        cv.circle(target, (center_X, center_Y),
                  radius, MAGENTA, 2)

        for i, rr in enumerate(rings_radii):
            if (center_X - ring_center_X)**2 + \
               (center_Y - ring_center_Y)**2 <= rr**2:
                hitpoints += (10 - i)
                print(f"shot #0{index + 1} -> points: {10 - i}")
                break

    print(f"total: {hitpoints}")

    return len(hits), target
