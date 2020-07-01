
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
import math

# Resize if it's too big. Limit height to 800px.
def scaleToLimit(image, rowLimit):
    rat = rowLimit / image.shape[0]
    if (rat < 1):
        return cv.resize(image, None, fx=rat, fy=rat)
    return image;

# Protocol: read -> scale -> blur -> canny -> hough transform
image = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)
image = scaleToLimit(image, 400)
orig = np.copy(image)
c_orig = cv.cvtColor(orig, cv.COLOR_GRAY2BGR)
image = cv.GaussianBlur(image, (5,5), 0)
mean = np.mean(image)
image = cv.Canny(image, mean/3, mean*4/3)

cv.imshow("Canny Output", image)
cv.waitKey(0)

# Hough. Just Hough.
lines = cv.HoughLines(image, 1, np.pi / 180, 150, None, 0, 0)
# convert HoughLines output to graphable lines.
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv.line(c_orig, pt1, pt2, (0,0,255), 3, cv.LINE_AA)

cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", c_orig)
cv.waitKey(0)

# OLD WAY
#image = imutils.rotate_bound(image, 30) rotateMatrix(image, 30)
# cv.imshow("original", image)
# cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# cannyOutput = cv.canny(image)
# matchedLines = cannyOutput[0]
# dominantAngle = cannyOutput[1]
# linesRotated = cv.rotateMatrix(matchedLines, dominantAngle)
