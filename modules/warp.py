import cv2
import numpy as np

def calcDistance(p1: tuple[int | float, int | float], p2: tuple[int | float, int | float]) -> float:
    """ 
    Calculate distance between two points
    Input:
        p1, p2: coordinate of point (x, y)
    Output:
        Distance (type: float)
    """
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**(0.5)

def warpDocument(img: np.ndarray, lt: tuple[int, int], lb: tuple[int, int], rt: tuple[int, int], rb: tuple[int, int]) -> np.ndarray:
    """
    Warp the document in the given image using four given points (corner of document).
    Assumption:
        1. Document is rectangle. (Which means, four angle of corners are all right angle.)
        2. Document in image is linearly warped. (Four edge are all straight line in image.)
        3. Four corners is given at right position. (This can be improved...??)
    Input:
        img: numpy array of image.
        lt: left top coordinate of image (x, y)
        lb: left bottom coordinate of image (x, y)
        rt: right top coordinate of image (x, y)
        rb: right bottom coordinate of image (x, y)
    Output:
        numpy array of warped document image.
    """

    """
    Step for restoring document:
        1. 3D Rotation: Make warped document to rectangle (parallel to each other, and all 4 corners are normal)
             1  0  0
             0  1  0
            Rx Ry Rz
        2. 2D Translation: Move lt (left top) to (0, 0)
             1  0 tx
             0  1 ty
             0  0  1
        3. 2D Rotation: Make rectangle's edges parallel to axis x and axis y
             cos_a -sin_a      0
             sin_a  cos_a      0
                 0      0      1
        4. Uniform scale: Make it to proper size
            S 0 0
            0 S 0 
            0 0 1
    """
    pass


def warpDocumentNaive(
        img: np.ndarray, lt: tuple[int, int], lb: tuple[int, int], rt: tuple[int, int], rb: tuple[int, int]
    ) -> np.ndarray:
    """
    Warp the document in the given image using four given points (corner of document).
    Assumption:
        1. Document is rectangle. (Which means, four angle of corners are all right angle.)
        2. Document in image is linearly warped. (Four edge are all straight line in image.)
        3. Four corners is given at right position. (This can be improved...??)
        4. Camera has small 3d rotation with normal vector of document. (Naive approch)
    Input:
        img: numpy array of image.
        lt: left top coordinate of image (x, y)
        lb: left bottom coordinate of image (x, y)
        rt: right top coordinate of image (x, y)
        rb: right bottom coordinate of image (x, y)
    Output:
        numpy array of warped document image.
    """

    """
    Step for restoring document:
        1. 2D Translation: Move lt (left top) to (0, 0)
             1  0 tx
             0  1 ty
             0  0  1
        2. Pick width, height to maximum width and maximum height from the warped document
        3. Set 4 corners based on step 2's width and height, and get transformation matrix
        4. Warp with calculated transformation matrix
    """

    # 1. 2D Translation: Move lt (left top) to (0, 0)
    tx = -lt[0]; ty = -lt[1]
    ltTransformed = (0, 0)
    lbTransformed = ((lb[0] + tx), (lb[1] + ty))
    rtTransformed = ((rt[0] + tx), (rt[1] + ty))
    rbTransformed = ((rb[0] + tx), (rb[1] + ty))

    # 2. Pick width, height to maximum width and maximum height from the warped document
    originalWidth = max(calcDistance(ltTransformed, rtTransformed), calcDistance(lbTransformed, rbTransformed))
    originalHeight = max(calcDistance(ltTransformed, lbTransformed), calcDistance(rtTransformed, rbTransformed))
 
    # 3. Set 4 corners based on step 2's width and height, and get transformation matrix
    ltOriginal = (0, 0)
    lbOriginal = (0, originalHeight) 
    rtOriginal = (originalWidth, 0)
    rbOriginal = (originalWidth, originalHeight)
    transMat = cv2.getPerspectiveTransform(
        src = np.array([list(lt), list(lb), list(rt), list(rb)]).astype(np.float32),
        dst = np.array([list(ltOriginal), list(lbOriginal), list(rtOriginal), list(rbOriginal)]).astype(np.float32),
    )

    # 4. Warp with calculated transformation matrix
    warpped_img = cv2.warpPerspective(img, transMat, (int(np.ceil(originalWidth)), int(np.ceil(originalHeight))))

    return warpped_img


if __name__ == "__main__":
    img = cv2.imread('test/image/ProgrammingLanguage.jpg', 1).astype(np.float32)
    warpped_img = warpDocumentNaive(
        img,
        lt=(155, 1323),
        lb=(1318, 3726),
        rt=(1627, 804),
        rb=(2979, 2641),
    )
    cv2.imwrite('test/warp_result/warpped.jpg', warpped_img)
