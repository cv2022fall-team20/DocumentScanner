import glob

import cv2
import numpy as np

from modules.detect_corners import (MAX_RESIZED_DIMENSION, detect_corners,
                                    resize_image)
from modules.warp import warpDocumentSimple


def scan_document(image: np.ndarray) -> np.ndarray:
    """
    Detect and warp a document in an image.
    """
    corners = detect_corners(image)
    warped_image = warpDocumentSimple(image, corners[0], corners[3],
                                      corners[1], corners[2])
    return warped_image


def main():
    """
    Run document detection and warping on each image in the test images
    directory. Press any key to continue to the next image.
    """
    for image_path in glob.glob('test/image/*'):
        image = cv2.imread(image_path)
        document_image = scan_document(image)
        resized_image, _ = resize_image(document_image, MAX_RESIZED_DIMENSION)
        cv2.imshow('Document', resized_image)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
