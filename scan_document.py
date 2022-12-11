import glob

import cv2
import numpy as np

from modules import MAX_RESIZED_DIMENSION, detect_corners, resize_image
from modules import warpDocumentSimple, warpDocumentGradientDescent
from modules import post_process_high_contrast, post_process_laplacian


def scan_document(image: np.ndarray, handleWarp: str, paperType: str | None = None) -> np.ndarray:
    """
    Detect and warp a document in an image.
    """
    corners: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] = detect_corners(image)
    if handleWarp == "simple":
        warped_image = warpDocumentSimple(image, corners[0], corners[3],
                                        corners[1], corners[2], paperType)
    elif handleWarp == "gradient":
        warped_image = warpDocumentGradientDescent(image, corners[0], corners[3],
                                        corners[1], corners[2])
    else:
        raise Exception("Not appropriate args")
    return warped_image


def main():
    """
    Run document detection and warping on each image in the test images
    directory. Press any key to continue to the next image.
    """
    for image_path in glob.glob('test/image/*'):
        # Simple algorithm
        image = cv2.imread(image_path)
        document_image = scan_document(image, "simple")
        resized_image, _ = resize_image(document_image, MAX_RESIZED_DIMENSION)
        high_contrasted_image = post_process_high_contrast(resized_image)
        laplacian_image = post_process_laplacian(resized_image)
        cv2.imwrite(image_path.replace("image", "final_result/high_contrast/simple"), high_contrasted_image)
        cv2.imwrite(image_path.replace("image", "final_result/laplacian/simple"), laplacian_image)
        # cv2.imshow('Document Simple (HC)', high_contrasted_image)
        # cv2.waitKey(0)
        # cv2.imshow('Document Simple (DoG)', laplacian_image)
        # cv2.waitKey(0)
        

        # Simple algorithm with given ratio
        document_image = scan_document(image, "simple", 'A')
        resized_image, _ = resize_image(document_image, MAX_RESIZED_DIMENSION)
        high_contrasted_image = post_process_high_contrast(resized_image)
        laplacian_image = post_process_laplacian(resized_image)
        cv2.imwrite(image_path.replace("image", "final_result/high_contrast/ratio_given"), high_contrasted_image)
        cv2.imwrite(image_path.replace("image", "final_result/laplacian/ratio_given"), laplacian_image)
        # cv2.imshow('Document Ratio Given (HC)', high_contrasted_image)
        # cv2.waitKey(0)
        # cv2.imshow('Document Ratio Given (DoG)', laplacian_image)
        # cv2.waitKey(0)

        # Gradient Descent algorithm
        document_image = scan_document(image, "gradient")
        resized_image, _ = resize_image(document_image, MAX_RESIZED_DIMENSION)
        high_contrasted_image = post_process_high_contrast(resized_image)
        laplacian_image = post_process_laplacian(resized_image)
        cv2.imwrite(image_path.replace("image", "final_result/high_contrast/gradient"), high_contrasted_image)
        cv2.imwrite(image_path.replace("image", "final_result/laplacian/gradient"), laplacian_image)
        # cv2.imshow('Document Gradient (HC)', high_contrasted_image)
        # cv2.waitKey(0)
        # cv2.imshow('Document Gradient (DoG)', laplacian_image)
        # cv2.waitKey(0)


if __name__ == '__main__':
    main()
