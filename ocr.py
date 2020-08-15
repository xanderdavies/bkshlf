# OCR

# %% imports
from imutils import rotate_bound
from imutils.object_detection import non_max_suppression
from PIL import Image, ImageDraw
import pytesseract
from pytesseract import Output
from matplotlib.image import imread
import scipy.misc
from scipy.ndimage.morphology import binary_dilation
from detectron2.data import detection_utils
import detectron2
import re
import numpy as np
import cv2

# %% settings
min_confidence = .5 # PLAY WITH
long_side = 800     # PLAY WITH
padding = 0.06      # PLAY WITH
east_path = "./shelves/frozen_east_text_detection.pb"
classes = ["book_spine", "inc_spine", "no_text", "book_cover", "inc_cover"]

# %% cropper function - add buffer, fix straighten
# https://github.com/facebookresearch/detectron2/issues/984 was helpful
def cropper(org_image_path, out_file_dir, predictor):

    # rotation helper
    def get_height(mask_array):
        top_of_mask = mask.shape[0]
        bottom_of_mask = 0
        no_mask_yet = True
        for row_number, row in enumerate(mask_array):
            if max(row.flatten()) == 0:
                if no_mask_yet:
                    top_of_mask = mask.shape[0] - row_number
            else:
                no_mask_yet = False
                bottom_of_mask = mask.shape[0] - row_number
        return (top_of_mask - bottom_of_mask)

    # resizing helper — can be improved, right now run twice
    def get_new_dims(opened_image):
        (origH, origW) = opened_image.shape[:2]
        # scale based on long_side provided
        if origH > origW:
            short_side = int(((long_side/origH)*origW//32 + 1)*32)
            (newW, newH) = (short_side, long_side)
        else:
            short_side = int(((long_side/origW)*origH//32 + 1)*32)
            (newW, newH) = (long_side, short_side)
        return (newW, newH)

    # open image, make spine predictions
    filename = (org_image_path.split("/")[-1]).split(".")[0]
    img = cv2.imread(org_image_path)
    outputs = predictor(img)
    instances = outputs["instances"].to('cpu')

    # bounding boxes
    boxes = instances.pred_boxes
    if isinstance(boxes, detectron2.structures.boxes.Boxes):
        boxes = boxes.tensor.numpy()
    else:
        boxes = np.asarray(boxes)

    # labels
    labels = [classes[i] for i in instances.pred_classes]

    # masks
    mask_array = instances.pred_masks.numpy()  # pred masks are now nd-numpy arrays
    num_instances = mask_array.shape[0]  # number of books/created images
    mask_array = np.moveaxis(mask_array, 0, -1)
    mask_array_instance = []  # initialize instances list

    # initialize zero image
    img = imread(str(org_image_path))
    output = np.zeros_like(img)
    output_file_names = []  # initialize file names list

    for i in range(num_instances):
        if labels[i] == "book_spine":
            mask_array_instance.append(mask_array[:, :, i:(i+1)])
            mask = np.array(mask_array_instance[i], dtype=bool)
            dilated_mask = binary_dilation(mask, iterations=10)

            # KEY LINE - if not mask array, then 255 (white), else copy from img
            output = np.where(dilated_mask == False, 0, img)
            im = Image.fromarray(output)
            image = np.array(im.crop(boxes[i]))

            # resize one
            new_dims = get_new_dims(image)
            image = cv2.resize(image, new_dims)

            # rotate — TO DO optimization by MAX
            best_angle = [0, get_height(image)]
            for t in range(180):
                dst = rotate_bound(image, -t)
                height = get_height(dst)
                if height < best_angle[1]:
                    best_angle = [t, height]
                    best_image = dst

            # resize two
            newer_dims = get_new_dims(best_image)
            best_image = cv2.resize(best_image, newer_dims)

            # # IF WANT TO SHOW IMAGE
            # cv2.imshow("spine", best_image)
            # cv2.waitKey()

            # save and update file names list
            output_file_names.append(f"{out_file_dir}/{filename}_{i}.jpg")
            image = cv2.cvtColor(best_image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image.save(f"{out_file_dir}/{filename}_{i}.jpg")
            print(f"Image {i} done, rescaled to {newer_dims}")

    return output_file_names


# %% read image function
import matplotlib.pyplot as plt
import keras_ocr
import re

def image_reader_2(image_path):
    # !pip install keras-ocr

    # keras-ocr will automatically download pretrained
    # weights for the detector and recognizer.
    pipeline = keras_ocr.pipeline.Pipeline()

    # Get a set of three example images
    image = keras_ocr.tools.read(image_path)

    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples.
    predictions = pipeline.recognize([image])[0]

    # Get text
    text = []
    for prediction in predictions:
        word = prediction[0]
        if re.search("..", word) != None:
            text.append(word)

    text = ' '.join(text)
    print(text)
    # Plot the predictions
    keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=None)
    plt.show()

    return text

# %% decode_predictions function (helper for image_reader)
def decode_predictions(scores, geometry):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < min_confidence:
                continue
            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)

def image_reader(org_image_path):
    image = cv2.imread(org_image_path)
    (H, W) = image.shape[:2]
    print(f"height: {H} width: {W}")
    # cv2.imshow("image_to_be_read", image)
    # cv2.waitKey()

    # define the two output layer names for the EAST detector model that
    # we are interested in -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    net = cv2.dnn.readNet(east_path)

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)  # ISSUE HERE

    # decode the predictions, then apply non-maxima suppression to
    # suppress weak, overlapping bounding boxes
    (rects, confidences) = decode_predictions(scores, geometry)
    if rects == []:
        print("failed to locate text")

    orig = image.copy()
    (origH, origW) = image.shape[:2]

    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # initialize the list of results
    results = []

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # add padding
        dX = int((endX - startX) * padding)
        dY = int((endY - startY) * padding)
        # apply padding to each side of the bounding box, respectively
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(origW, endX + (dX * 2))
        endY = min(origH, endY + (dY * 2))
        # extract the actual padded ROI
        roi = orig[startY:endY, startX:endX]

        if dY > dX:
            roi = rotate_bound(roi, 90)

        cv2.imshow("roi", roi)
        cv2.waitKey()

        # in order to apply Tesseract v4 to OCR text we must supply
        # (1) a language, (2) an OEM flag of 4, indicating that the we
        # wish to use the LSTM neural net model for OCR, and finally
        # (3) an PSM value, in this case, 7 which implies that we are
        # treating the ROI as a single line of text
        config = ("-l eng --oem 1 --psm 6")
        text = pytesseract.image_to_string(roi, config=config)
        # add the bounding box coordinates and OCR'd text to the list
        # of results
        results.append(((startX, startY, endX, endY), text))

    # sort the results bounding box coordinates from top to bottom
    results = sorted(results, key=lambda r: r[0][1])

    # %% show results, comment out for final pipeline
    for ((startX, startY, endX, endY), text) in results:
        # display the text OCR'd by Tesseract
        print("OCR TEXT")
        print("========")
        print("{}\n".format(text))
        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV, then draw the text and a bounding box surrounding
        # the text region of the input image
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        output = orig.copy()
        cv2.rectangle(output, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(output, text, (startX, startY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        # show the output image
        cv2.imshow("output", output)
        cv2.waitKey()

    book_text = []
    for word in results:
        book_text.append(word[1])
    book_text = ' '.join(book_text)

    return book_text


# NONESSENTIAL (STILL USEFUL) BELOW


# %% possible installs
# !sudo apt install tesseract-ocr
# !sudo add-apt-repository ppa:alex-p/tesseract-ocr
# !sudo apt-get update
# !sudo apt install tesseract-ocr
# !tesseract -v # MUST BE V4
# !pip install pillow
# !pip install pytesseract
# !pip install imutils
# !tesseract --help-l
# !tesseract --help-oem
# !tesseract --help-psm # 6 or 7? TRY CHANGING THIS
# !pip install opencv-contrib-python


# # %% example
# src = '/Users/xanderdavies/Desktop/bkshlf/shelf/shelves/val/ideal.JPG'
# out_folder = '/Users/xanderdavies/Desktop/bkshlf/shelf/shelves/output_images'
#
# # CROP BY PREDICTIONS
# # won't run because no predictor here
# output_file_names = cropper(src, out_folder, predictor)
# ex_image_path = output_file_names[3]
# print(image_reader(ex_image_path))
#
#
# # %% show results, not incorporated yet
# for ((startX, startY, endX, endY), text) in results:
#   # display the text OCR'd by Tesseract
#   print("OCR TEXT")
#   print("========")
#   print("{}\n".format(text))
#   # strip out non-ASCII text so we can draw the text on the image
#   # using OpenCV, then draw the text and a bounding box surrounding
#   # the text region of the input image
#   text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
#   output = orig.copy()
#   cv2.rectangle(output, (startX, startY), (endX, endY),
#     (0, 0, 255), 2)
#   cv2.putText(output, text, (startX, startY - 20),
#     cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
#   # show the output image
#   cv2.imshow(output)
#   cv2.waitKey()

################
#
# def cropper(org_image_path, out_file_dir, pred):
#     filename = (org_image_path.split("/")[-1]).split(".")[0]
#     img = cv2.imread(org_image_path)# detection_utils.read_image(org_image_path, format="BGR")
#     outputs = pred(img)
#     instances = outputs["instances"].to('cpu')
#
#     # bounding boxes
#     boxes = instances.pred_boxes
#     if isinstance(boxes, detectron2.structures.boxes.Boxes):
#         boxes = boxes.tensor.numpy()
#     else:
#         boxes = np.asarray(boxes)
#
#     # labels
#     labels = [classes[i] for i in instances.pred_classes]
#
#     # masks
#     mask_array = instances.pred_masks.numpy() # pred masks are now nd-numpy arrays
#     num_instances = mask_array.shape[0] # number of books/created images
#     mask_array = np.moveaxis(mask_array, 0, -1)
#     mask_array_instance = [] # initialize instances list
#
#     # initialize zero image
#     img = imread(str(org_image_path))
#     output = np.zeros_like(img)
#     output_file_names = [] # initialize file names list
#
#     for i in range(num_instances):
#         # improve this by calculating minimum distance between top and bottom points
#         if labels[i] == "book_spine":
#             mask_array_instance.append(mask_array[:, :, i:(i+1)])
#             output = np.where(mask_array_instance[i] == False, 0, img) # KEY LINE - if not mask array, then 255 (white), else copy from img
#             # im = rotate_bound(output, 270) # rotate 270
#             im = Image.fromarray(output)
#             image = im.crop(boxes[i])
#
#             image = np.array(image)
#             # resize done here instead
#             orig = image.copy()
#             (origH, origW) = image.shape[:2]
#             # correctly scale based on long_side provided
#             if origH > origW:
#                 short_side = int(((long_side/origH)*origW//32 + 1)*32)
#                 (newW, newH) = (short_side, long_side)
#             else:
#                 short_side = int(((long_side/origW)*origH//32 + 1)*32)
#                 (newW, newH) = (long_side, short_side)
#
#             rW = origW / float(newW)
#             rH = origH / float(newH)
#             # resize the image and grab the new image dimensions
#             image = cv2.resize(image, (newW, newH))
#
#
#             # save and update file names list
#             output_file_names.append(f"{out_file_dir}/{filename}_{i}.jpg")
#             image = Image.fromarray(image)
#             image.save(f"{out_file_dir}/{filename}_{i}.jpg")
#
#     return output_file_names
