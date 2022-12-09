import cv2
import numpy as np

# Hyperparameters.
GAUSSIAN_BLUR_KERNEL_SIZE = (5, 5)
ADAPTIVE_THRESHOLD_BLOCK_SIZE = 11
ADAPTIVE_THRESHOLD_C = 2
INITIAL_EPSILON_MULTIPLIER = 0.01
EPSILON_MULTIPLIER_INCREMENT = 0.01
MAX_EPSILON_MULTIPLIER = 0.1
MIN_AREA_PROPORTION = 0.1


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
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(grayscale_image,
                                     GAUSSIAN_BLUR_KERNEL_SIZE, 0)
    thresholded_image = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, ADAPTIVE_THRESHOLD_BLOCK_SIZE, ADAPTIVE_THRESHOLD_C)
    contours, _ = cv2.findContours(
        thresholded_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sort the contours by length.
    contours = sorted(contours,
                      key=lambda contour_: cv2.arcLength(contour_, False),
                      reverse=True)
    # The epsilon parameter in cv2.approxPolyDP is the maximum distance between
    # a point on the contour and the approximated contour. Start with a small
    # epsilon value and increase it until an approximated contour with four
    # points is found.
    epsilon_multiplier = INITIAL_EPSILON_MULTIPLIER
    image_area = image.shape[0] * image.shape[1]
    while epsilon_multiplier <= MAX_EPSILON_MULTIPLIER:
        for contour in contours:
            contour_length = cv2.arcLength(contour, False)
            approximated_contour = cv2.approxPolyDP(
                contour, epsilon_multiplier * contour_length, True)
            # Check if the approximated contour is a convex quadrilateral with
            # a large enough area.
            if (len(approximated_contour) == 4
                    and cv2.isContourConvex(approximated_contour)
                    and cv2.contourArea(approximated_contour)
                    >= MIN_AREA_PROPORTION * image_area):
                corners = tuple(map(tuple, approximated_contour.reshape(4, 2)))
                sorted_corners = sort_corner_coordinates(corners)
                return sorted_corners
        else:
            epsilon_multiplier += EPSILON_MULTIPLIER_INCREMENT
    else:
        raise RuntimeError('Failed to detect document.')


def main():
    image = cv2.imread('../test/image/ProgrammingLanguageSmall.jpg')
    corners = detect_corners(image)
    # Draw the edges of the document.
    for i in range(4):
        cv2.line(image, corners[i], corners[(i + 1) % 4], (0, 0, 255), 2)
    # Label the corners.
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1
    text_thickness = 2
    text_size, _ = cv2.getTextSize('0', font, font_scale, text_thickness)
    for i, corner in enumerate(corners):
        text_origin = (corner[0] - text_size[0] // 2,
                       corner[1] + text_size[1] // 2)
        cv2.putText(image, str(i), text_origin, font, font_scale, (0, 255, 0),
                    text_thickness, cv2.LINE_AA)
    cv2.imshow('Detected document', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
