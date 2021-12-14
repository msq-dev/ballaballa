import sys
import imghdr
import cv2 as cv
from balla.img import BallaImage
from balla.detectors import NoHitsError, NoTargetError, detect_target, detect_hits


def main() -> None:
    """Detect hits on target

    Parameters
    ----------
    args[0] : str
              Path to image of target
    """

    args = sys.argv[1:]

    if not len(args) or not imghdr.what(args[0]):
        print("Please provide an image")
        return

    IMG_SRC = args[0]
    target = BallaImage(cv.imread(IMG_SRC), "Target")
    target.blur()

    try:
        rect = detect_target(target)
    except NoTargetError:
        return

    warped = BallaImage(target.img, "Warped")
    warped.warp(rect)

    try:
        detect_hits(warped)
        # no_of_hits, target = detect_hits(warped)
    except NoHitsError:
        return

    # show(target, window_title=f"HITS: {no_of_hits}")


if __name__ == "__main__":
    main()
