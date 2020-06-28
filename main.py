
  # The plan:
  # 1. Detect edges
  # 2. Filter out short edges
  # (TODO 3. Split image into shelves)
  # 4. Calculate dominant angle
  # 5. Segment spines
  # 6. Crop spines
  # 7. Isolate text features
  # 8. Send to OCR
  # 9. Interface.


import cv2 as cv
import sys
import numpy as np
import imutils


def scaleToLimit(image, rowLimit):
    # Resize if it's too big. Limit height to 800px.
    rat = rowLimit / image.shape[0]
    if (rat < 1):
        return cv.resize(image, None, fx=rat, fy=rat)
    return image;

image = cv.imread(sys.argv[1], cv.IMREAD_COLOR)
image = scaleToLimit(image, 400)

def rotateMatrix(im, degrees):
    rows = im.shape[0]
    cols = im.shape[1]
    M = cv.getRotationMatrix2D((cols/2,rows/2),degrees,1)
    rads = degrees / 180 * np.pi
    new_size = ((int) (np.sin(rads) * rows + np.cos(rads) * cols), (int) (np.cos(rads) * rows + np.sin(rads) * cols))
    print(new_size)
    print(np.cos(rads), np.sin(rads), rows, cols)
    dst = cv.warpAffine(im, M, new_size)
    return dst

image = imutils.rotate_bound(image, 30) #rotateMatrix(image, 30)
cv.imshow("original", image)
# cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# cannyOutput = cv.canny(image)
# matchedLines = cannyOutput[0]
# dominantAngle = cannyOutput[1]
# linesRotated = cv.rotateMatrix(matchedLines, dominantAngle)

cv.waitKey(0)
