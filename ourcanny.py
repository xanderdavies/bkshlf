import cv2 as cv
import numpy as np

def median(grayImage):
    return numpy.mean(grayImage)

# Please pass in image in greyscale, ie 2D numpy array
# If it's in color, it'll be a 3D array
def traceEdge(image, row, col):
    stack = [(row, col)]
    neighbors = []
    while(len(stack) > 0):
        x, y = stack.pop()
        neighbors.append((x,y));
        for i in range(max(0, x - 1), min(image.shape[0], x + 2)):
            for j in range(max(0, y - 1), min(image.shape[1], y + 2)):
                if(image[i,j] > 0):
                    stack.append((i,j))
                    image[i,j] = 0
    return neighbors


def bin(theta, nbins):
    return (theta / ((cv.CV_2PI) / nbins))

# Find the angle of a component, return an int in range [0, 180],
# where 0 means vertical, 90 means horizontal.
def componentAngle(component):
    # Use opencv fitLine to obtain the vector that fits the component best.
    # Vec4f outLine; ??
    outLine = cv.fitLine(component, cv.CV_DIST_L2, 0, 0.01, 0.01)
    angle = (int) (180 / cv.CV_PI * np.arctan(outLine[0] / outLine[1]))
    return (angle if angle >= 0 else angle + 180)


# Find the difference between angles a and b, which are degrees in [0, 180].
# Result is in [0, 90].

def angleDiff(a, b):
    diff = abs(a - b)
    return (diff if diff < 90 else 180 - diff)

def rotateMatrix(im, degrees):
    rows = im.shape[0]
    cols = im.shape[1]
    M = cv.getRotationMatrix2D((cols/2,rows/2),degrees,1)
    dst = warpAffine(im, M, (cos(degrees) * (rows + cols), sin(degrees) * (rows + cols)))
    return dst

def connectedComponents(grayImage):
    cpy = grayImage.clone()
    for i in range(cpy.rows):
        row =

"""
std::vector<std::vector<Point>> connectedComponents(const Mat& grayImage) {
    // Construct a copy of the image for bookkeeping.
    std::vector<std::vector<Point>> edgesOutput;
    Mat cpy = grayImage.clone();
    for (int i = 0; i < cpy.rows; i++) {
        const uchar *row = cpy.ptr<uchar>(i);
        for (int j = 0; j < cpy.cols; j++) {
            if (row[j] > 0) {
                edgesOutput.push_back(traceEdge(cpy, i, j));
            }
        }
    }
    return edgesOutput;
}
"""
