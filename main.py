
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


def scaleToLimit(image, rowLimit):
    # Resize if it's too big. Limit height to 800px.
    rat = rowLimit / image.shape[0]
    if (rat < 1):
        return cv.resize(image, None, fx=rat, fy=rat)
    return image;

image = cv.imread(sys.argv[1], cv.IMREAD_COLOR)
image = scaleToLimit(image, 800)

def rotateMatrix(im, degrees):
    rows = im.shape[0]
    cols = im.shape[1]
    M = cv.getRotationMatrix2D((cols/2,rows/2),degrees,1)
    dst = cv.warpAffine(im, M, ((int) (np.cos(degrees) * (rows + cols)), (int) (np.sin(degrees) * (rows + cols))))
    return dst

image = rotateMatrix(image, 30)
cv.imshow("original", image)
# cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# cannyOutput = cv.canny(image)
# matchedLines = cannyOutput[0]
# dominantAngle = cannyOutput[1]
# linesRotated = cv.rotateMatrix(matchedLines, dominantAngle)

cv.waitKey(0)
