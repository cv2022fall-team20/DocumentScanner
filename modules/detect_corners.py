import cv2
import numpy as np

# Hyperparameters.
MAX_RESIZED_DIMENSION = 960
GAUSSIAN_BLUR_KERNEL_SIZE = 5
ADAPTIVE_THRESHOLD_BLOCK_SIZE = 11
ADAPTIVE_THRESHOLD_C = 2
CANNY_LOWER_THRESHOLD = 10
CANNY_UPPER_THRESHOLD = 100
DILATION_KERNEL_SIZE = 3
INITIAL_EPSILON_MULTIPLIER = 0.01
EPSILON_MULTIPLIER_INCREMENT = 0.01
MAX_EPSILON_MULTIPLIER = 0.1
MIN_AREA_PROPORTION = 0.1
MIN_IMAGE_BORDER_GAP = 2


def resize_image(image: np.ndarray, max_dimension: int) -> tuple[np.ndarray,
                                                                 float]:
    """
    Resize an image so that its largest dimension is equal to max_dimension.
    Return the resized image and the scale factor used to resize the image.
    """
    resize_scale = max_dimension / max(image.shape[:2])
    resized_image = cv2.resize(image, None, fx=resize_scale, fy=resize_scale)
    return resized_image, resize_scale


def preprocess_image(image: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Preprocess an image by converting it to grayscale, resizing it, and
    blurring it. Return the preprocessed image and the scale factor used to
    resize the image.
    """
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image, resize_scale = resize_image(grayscale_image,
                                               MAX_RESIZED_DIMENSION)
    blurred_image = cv2.GaussianBlur(
        resized_image, (GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_KERNEL_SIZE),
        0)
    return blurred_image, resize_scale


def apply_adaptive_thresholding(preprocessed_image: np.ndarray) -> np.ndarray:
    """
    Apply adaptive thresholding to a preprocessed image.
    """
    thresholded_image = cv2.adaptiveThreshold(
        preprocessed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, ADAPTIVE_THRESHOLD_BLOCK_SIZE, ADAPTIVE_THRESHOLD_C)
    return thresholded_image


def apply_canny_edge_detector(preprocessed_image: np.ndarray) -> np.ndarray:
    """
    Apply the Canny edge detector to a preprocessed image and dilate the edges
    to eliminate small gaps. Return the detected edges as a binary image.
    """
    canny_edges = cv2.Canny(preprocessed_image, CANNY_LOWER_THRESHOLD,
                            CANNY_UPPER_THRESHOLD)
    dilation_kernel = np.ones((DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE),
                              np.uint8)
    dilated_edges = cv2.dilate(canny_edges, dilation_kernel)
    return dilated_edges


def find_document_contour(binarized_image: np.ndarray) -> np.ndarray:
    """
    Find the contour of a document in a binarized image. Return the contour as
    an array of shape (4, 1, 2), or None if no document contour was found.
    """
    contours, _ = cv2.findContours(binarized_image, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    # Sort the contours by length.
    contours = sorted(contours,
                      key=lambda contour_: cv2.arcLength(contour_, False),
                      reverse=True)
    # The epsilon parameter in cv2.approxPolyDP is the maximum distance between
    # a point on the contour and the approximated contour. Start with a small
    # epsilon value and increase it until an approximated contour with four
    # points is found.
    epsilon_multiplier = INITIAL_EPSILON_MULTIPLIER
    resized_image_height, resized_image_width = binarized_image.shape[:2]
    resized_image_area = resized_image_height * resized_image_width
    while epsilon_multiplier <= MAX_EPSILON_MULTIPLIER:
        for contour in contours:
            contour_length = cv2.arcLength(contour, False)
            approximated_contour = cv2.approxPolyDP(
                contour, epsilon_multiplier * contour_length, True)
            # Check if the contour is a convex quadrilateral with a large
            # enough area.
            if (len(approximated_contour) != 4
                    or not cv2.isContourConvex(approximated_contour)
                    or cv2.contourArea(approximated_contour)
                    < MIN_AREA_PROPORTION * resized_image_area):
                continue
            # Check if any of the points is too close to the image border.
            if any(point[0][0] < MIN_IMAGE_BORDER_GAP
                   or point[0][0] >= resized_image_width - MIN_IMAGE_BORDER_GAP
                   or point[0][1] < MIN_IMAGE_BORDER_GAP
                   or point[0][1] >= (resized_image_height
                                      - MIN_IMAGE_BORDER_GAP)
                   for point in approximated_contour):
                continue
            return approximated_contour
        else:
            epsilon_multiplier += EPSILON_MULTIPLIER_INCREMENT
    else:
        return None


def sort_corner_coordinates(corners: tuple[tuple[int, int], tuple[int, int],
                                           tuple[int, int], tuple[int, int]]) \
        -> tuple[tuple[int, int], tuple[int, int],
                 tuple[int, int], tuple[int, int]]:
    """
    Sort the coordinates of the detected corners in clockwise order starting
    from the top-left corner.
    """
    # Calculate the center coordinates of the detected corners.
    center_x = sum(corner[0] for corner in corners) / 4
    center_y = sum(corner[1] for corner in corners) / 4
    # Sort the corners by their angle relative to the center. The y-axis is
    # inverted in the image coordinates.
    corners = sorted(corners,
                     key=lambda corner: np.arctan2(corner[1] - center_y,
                                                   corner[0] - center_x))
    return corners


def detect_corners(image: np.ndarray) \
        -> tuple[tuple[int, int], tuple[int, int],
                 tuple[int, int], tuple[int, int]]:
    """
    Detect the four corners of a document in an image and return their
    coordinates. The coordinates are sorted in clockwise order starting from
    the top-left corner.
    """
    preprocessed_image, resize_scale = preprocess_image(image)
    # Try adaptive thresholding first. If it fails, use the Canny edge
    # detector.
    for binarization_function in (apply_adaptive_thresholding,
                                  apply_canny_edge_detector):
        binarized_image = binarization_function(preprocessed_image)
        document_contour = find_document_contour(binarized_image)
        if document_contour is None:
            continue
        corners = tuple(map(tuple, document_contour.reshape(4, 2)))
        sorted_corners = sort_corner_coordinates(corners)
        # Scale the coordinates of the corners back to the original image size.
        scaled_corners = tuple((round(corner[0] / resize_scale),
                                round(corner[1] / resize_scale))
                               for corner in sorted_corners)
        print(f'Document detected using '
              f'{" ".join(binarization_function.__name__.split("_")[1:])}.')
        return scaled_corners
    else:
        raise RuntimeError('Failed to detect document.')


def main():
    image = cv2.imread('../test/image/ProgrammingLanguage.jpg')
    corners = detect_corners(image)
    # Draw the edges of the document.
    max_image_dimension = max(image.shape[:2])
    edge_thickness = max_image_dimension // 400
    for i in range(4):
        cv2.line(image, corners[i], corners[(i + 1) % 4], (0, 0, 255),
                 edge_thickness)
    # Label the corners.
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = max_image_dimension / 1000
    text_thickness = max_image_dimension // 400
    text_size, _ = cv2.getTextSize('0', font, font_scale, text_thickness)
    for i, corner in enumerate(corners):
        text_origin = (corner[0] - text_size[0] // 2,
                       corner[1] + text_size[1] // 2)
        cv2.putText(image, str(i), text_origin, font, font_scale, (0, 255, 0),
                    text_thickness, cv2.LINE_AA)
    resized_image, _ = resize_image(image, MAX_RESIZED_DIMENSION)
    cv2.imshow('Detected document', resized_image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
