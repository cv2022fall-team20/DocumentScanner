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

def getWarpedPoint(p: tuple[int | float, int | float], transMat: np.ndarray) -> tuple[int | float, int | float]:
    pHomo = np.array([p[0], p[1], 1])
    pWarpedHomo = transMat @ pHomo
    return pWarpedHomo[0] / pWarpedHomo[2], pWarpedHomo[1] / pWarpedHomo[2]

def warpDocumentGradientDescent(
    img: np.ndarray, 
    leftTop: tuple[int, int], leftBottom: tuple[int, int], 
    rightTop: tuple[int, int], rightBottom: tuple[int, int],
    threshold: float = 1e-10, learning_rate: float = 1e-20, inertia: float = 0.9,
) -> np.ndarray:
    """
    Warp the document in the given image using four given points (corner of document).
    Assumption:
        1. Document is rectangle. (Which means, four angle of corners are all right angle.)
        2. Document in image is linearly warped. (Four edge are all straight line in image.)
        3. Four corners is given at right position. (This can be improved...??)
    Input:
        img: numpy array of image.
        leftTop: left top coordinate of image (x, y)
        leftBottom: left bottom coordinate of image (x, y)
        rightTop: right top coordinate of image (x, y)
        rightBottom: right bottom coordinate of image (x, y)
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
        3. 2D Rotation: Make rectangle's edges parallel to axis x and axis y
        4. Uniform scale: Make it to proper size
    """
    ######################################################################################################
    # 1. Find 3d warping which makes document edges to parallel
    # Find Intersect coordinate in homography coordinate of document edges.
    topLine = np.cross(
        np.array([leftTop[0], leftTop[1], 1]), np.array([rightTop[0], rightTop[1], 1])
    )
    bottomLine = np.cross(
        np.array([leftBottom[0], leftBottom[1], 1]), np.array([rightBottom[0], rightBottom[1], 1])
    )
    topBottomIntersect = np.cross(topLine, bottomLine)

    leftLine = np.cross(
        np.array([leftTop[0], leftTop[1], 1]), np.array([leftBottom[0], leftBottom[1], 1])
    )
    rightLine = np.cross(
        np.array([rightTop[0], rightTop[1], 1]), np.array([rightBottom[0], rightBottom[1], 1])
    )
    leftRightIntersect = np.cross(leftLine, rightLine)

    loss = 1
    rotateVector = np.array([0., 0., 1.])

    cnt = 0
    beforeGradLoss = np.zeros((3))
    while True:
        # Front pass
        topBottomIntersectWarpedScale: float = np.inner(topBottomIntersect, rotateVector)
        leftRightIntersectWarpedScale: float = np.inner(leftRightIntersect, rotateVector)
        scaleSquaredSum: float = np.square(topBottomIntersectWarpedScale) + np.square(leftRightIntersectWarpedScale)
        loss = np.sqrt(scaleSquaredSum)
        cnt += 1
        if cnt >= 1:
            print(loss)
            cnt = 0

        # Check
        if np.abs(loss) < threshold:
            break

        # Back propagation
        dLossOverdScaleSquaredSum = 1 / (2 * np.sqrt(scaleSquaredSum))
        dScaleSquaredSumOverdTopBottomIntersectWarpedScale = 2 * topBottomIntersectWarpedScale
        dScaleSquaredSumOverdLeftRightIntersectWarpedScale = 2 * leftRightIntersectWarpedScale
        gradLoss: np.ndarray = (
                (topBottomIntersect * dScaleSquaredSumOverdTopBottomIntersectWarpedScale)
                + (leftRightIntersect * dScaleSquaredSumOverdLeftRightIntersectWarpedScale)
            ) * dLossOverdScaleSquaredSum
        
        # Update
        beforeGradLoss = inertia * beforeGradLoss + gradLoss
        rotateVector -= beforeGradLoss * (learning_rate*loss) 

    
    rotate3dMatrix = np.vstack((np.array([
        [1, 0, 0],
        [0, 1, 0],
    ]), rotateVector))
    
    # Get 3d rotated points
    leftTop3DRot: tuple[int | float, int | float] = getWarpedPoint(leftTop, rotate3dMatrix)
    leftBottom3DRot = getWarpedPoint(leftBottom, rotate3dMatrix)
    rightTop3DRot = getWarpedPoint(rightTop, rotate3dMatrix)
    rightBottom3DRot = getWarpedPoint(rightBottom, rotate3dMatrix)

    # TODO: Assert normal

    ######################################################################################################
    # 2. Calculate 2D warping matrix (Translation & Rotation) using edge length
    # Get width and height
    width: float = np.sqrt(np.square(leftTop3DRot[0] - leftBottom3DRot[0]) + np.square(leftTop3DRot[1] - leftBottom3DRot[1]))
    height: float = np.sqrt(np.square(leftTop3DRot[0] - rightTop3DRot[0]) + np.square(leftTop3DRot[1] - rightTop3DRot[1]))

    # TODO: Assert same width & height
    # assert width == np.sqrt(np.square(rightTop3DRot[0] - rightBottom3DRot[0]) + np.square(rightTop3DRot[1] - rightBottom3DRot[1]))
    # assert height == np.sqrt(np.square(leftBottom3DRot[0] - rightBottom3DRot[0]) + np.square(leftBottom3DRot[1] - rightBottom3DRot[1]))

    # Set 4 destination corners based on width & height
    leftTopDst = (0, 0)
    leftBottomDst = (0, height)
    rightTopDst = (width, 0)
    rightBottomDst = (width, height)

    # Get 2D warping matrix
    warp2dMatrix: np.ndarray = cv2.getPerspectiveTransform(
        src = np.array([list(leftTop3DRot), list(leftBottom3DRot), list(rightTop3DRot), list(rightBottom3DRot)]).astype(np.float32),
        dst = np.array([list(leftTopDst), list(leftBottomDst), list(rightTopDst), list(rightBottomDst)]).astype(np.float32),
    )

    ######################################################################################################
    # 3. Warp Image based on calculated 3D rotation matrix and 2D warping matrix
    warpped_img: np.ndarray = cv2.warpPerspective(
        src = img, M = warp2dMatrix @ rotate3dMatrix, dsize = (int(np.ceil(width)), int(np.ceil(height)))
    )

    return warpped_img


def warpDocumentSimple(
        img: np.ndarray, lt: tuple[int, int], lb: tuple[int, int], rt: tuple[int, int], rb: tuple[int, int]
    ) -> np.ndarray:
    """
    Warp the document in the given image using four given points (corner of document).
    Assumption:
        1. Document is rectangle. (Which means, four angle of corners are all right angle.)
        2. Document in image is linearly warped. (Four edge are all straight line in image.)
        3. Four corners is given at right position. (This can be improved...??)
        4. Camera has small 3d rotation with normal vector of document. (Simplify approch)
    Input:
        img: numpy array of image.
        lt: left top coordinate of image (x, y)
        lb: left bottom coordinate of image (x, y)
        rt: right top coordinate of image (x, y)
        rb: right bottom coordinate of image (x, y)
    Output:
        numpy array of warped document image.
    """

    # 1. 2D Translation: Move lt (left top) to (0, 0).
    tx = -lt[0]; ty = -lt[1]
    ltTransformed = (0, 0)
    lbTransformed = ((lb[0] + tx), (lb[1] + ty))
    rtTransformed = ((rt[0] + tx), (rt[1] + ty))
    rbTransformed = ((rb[0] + tx), (rb[1] + ty))

    # 2. Pick width, height to maximum width and maximum height from the warped document.
    originalWidth = max(calcDistance(ltTransformed, rtTransformed), calcDistance(lbTransformed, rbTransformed))
    originalHeight = max(calcDistance(ltTransformed, lbTransformed), calcDistance(rtTransformed, rbTransformed))
 
    # 3. Set 4 corners based on step 2's width and height
    ltOriginal = (0, 0)
    lbOriginal = (0, originalHeight) 
    rtOriginal = (originalWidth, 0)
    rbOriginal = (originalWidth, originalHeight)

    # 4. Get transformation matrix based on corners on step 3.
    transMat = cv2.getPerspectiveTransform(
        src = np.array([list(lt), list(lb), list(rt), list(rb)]).astype(np.float32),
        dst = np.array([list(ltOriginal), list(lbOriginal), list(rtOriginal), list(rbOriginal)]).astype(np.float32),
    )

    # 5. Warp with calculated transformation matrix and return.
    warpped_img = cv2.warpPerspective(img, transMat, (int(np.ceil(originalWidth)), int(np.ceil(originalHeight))))

    return warpped_img


if __name__ == "__main__":
    # Small 3D rotation
    img = cv2.imread('test/image/ProgrammingLanguage.jpg', 1).astype(np.float32)
    warpped_img_simple = warpDocumentSimple(
        img,
        lt=(155, 1323),
        lb=(1318, 3726),
        rt=(1627, 804),
        rb=(2979, 2641),
    )
    cv2.imwrite('test/warp_result/warpped_straight_simple.jpg', warpped_img_simple)
    warpped_img_gd = warpDocumentGradientDescent(
        img,
        leftTop=(155, 1323),
        leftBottom=(1318, 3726),
        rightTop=(1627, 804),
        rightBottom=(2979, 2641),
    )
    cv2.imwrite('test/warp_result/warpped_straight_gd.jpg', warpped_img_gd)

    # Extream 3D rotation
    img_lean = cv2.imread('test/image/ProgrammingLanguageLean.jpg', 1).astype(np.float32)
    warpped_img_lean_simple = warpDocumentSimple(
        img_lean,
        lt=(97, 783),
        lb=(2253, 2806),
        rt=(1700, 288),
        rb=(3928, 1164),
    )
    cv2.imwrite('test/warp_result/warpped_lean_simple.jpg', warpped_img_lean_simple)
    warpped_img_lean_gd = warpDocumentGradientDescent(
        img_lean,
        leftTop=(97, 783),
        leftBottom=(2253, 2806),
        rightTop=(1700, 288),
        rightBottom=(3928, 1164),
    )
    cv2.imwrite('test/warp_result/warpped_lean_gd.jpg', warpped_img_lean_gd)
