# import torch, torchvision
# print(torch.__version__, torch.cuda.is_available())
#
# assert torch.__version__.startswith("1.6") # NOW 1.6, was 1.5

# Some basic setup:
# %% Setup detectron2 logger
import detectron2
# from detectron2.utils.logger import setup_logger
# setup_logger()

# %% import some common detectron2 utilities
# from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import DatasetCatalog, MetadataCatalog

# %% import some common libraries
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
import random
from datetime import datetime
import pickle
from pathlib import Path
from tqdm import tqdm

from detectron2.structures import BoxMode
#
# def get_spine_dicts(img_dir):
#     json_file = os.path.join(img_dir, "via_region_data.json")
#     with open(json_file) as f:
#         imgs_anns = json.load(f)
#
#     dataset_dicts = []
#     for idx, v in enumerate(imgs_anns.values()):
#         record = {}
#
#         filename = os.path.join(img_dir, v["filename"])
#         print(filename)
#
#         height, width = cv2.imread(filename).shape[:2]
#
#         record["file_name"] = filename
#         record["image_id"] = idx
#         record["height"] = height
#         record["width"] = width
#
#         annos = v["regions"]
#         objs = []
#         for annotation in annos:
#             anno = annotation["shape_attributes"]
#             px = anno["all_points_x"]
#             py = anno["all_points_y"]
#             poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
#             poly = [p for x in poly for p in x]
#
#             # specify the category_id to match with the class.
#             region_attributes = annotation["region_attributes"]["class"]
#
#             if "book_spine" in region_attributes:
#                 category_id = 0
#             elif "inc_spine" in region_attributes:
#                 category_id = 0
#             elif "no_text" in region_attributes:
#                 category_id = 0
#             elif "book_cover" in region_attributes:
#                 category_id = 1
#             elif "inc_cover" in region_attributes:
#                 category_id = 1
#
#             obj = {
#                 "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
#                 "bbox_mode": BoxMode.XYXY_ABS,
#                 "segmentation": [poly],
#                 "category_id": category_id,
#             }
#
#             if obj["category_id"] == 0: # ONLY LEARN SPINES
#                 objs.append(obj)
#         record["annotations"] = objs
#         dataset_dicts.append(record)
#     return dataset_dicts
#
# # %% the data has to be registered within detectron2
# for d in ["train", "val"]:
#     DatasetCatalog.register("shelf_" + d, lambda d=d: get_spine_dicts("/Users/xanderdavies/desktop/bkshlf/shelf/shelves/" + d))
#     MetadataCatalog.get("shelf_" + d).set(thing_classes=["book_spine", "not_spine"])
#     # shelf_metadata = MetadataCatalog.get("shelf_train").set(thing_classes=["book_spine", "not_spine"])
#
# shelf_metadata = MetadataCatalog.get("shelf_train")
#
# dataset_dicts = get_spine_dicts("/Users/xanderdavies/desktop/bkshlf/shelf/shelves/train")
#
# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=shelf_metadata, scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     cv2.imshow("img", out.get_image()[:, :, ::-1])
#     cv2.waitKey()
#
# """# Train"""
# %%
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("shelf_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.DEVICE = "cpu"
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 1000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # default: 512, was 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only one class rn

# trainer = DefaultTrainer(cfg)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#
# # # %% train
# # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# # trainer = DefaultTrainer(cfg)
# # trainer.resume_or_load(resume=False)
# # trainer.train()
# #
# # # Commented out IPython magic to ensure Python compatibility.
# # # look at training curves in tensorboard:
# # # %load_ext tensorboard
# # # %tensorboard --logdir output
#
# %%
"""# Test the model"""

# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") # NEED TO UNCOMMENT TO USE FRESHLY TRAINED MODEL

# if loading:
cfg.MODEL.WEIGHTS = '/Users/xanderdavies/desktop/bkshlf/shelf/shelves/saved_models/model_final.pth' # Dowloaded from /output in this collab

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  # GUESS — custom testing threshold for this model
cfg.DATASETS.TEST = ("shelf_val", )
from detectron2.modeling import build_model

predictor = DefaultPredictor(cfg)
# predictor = DefaultPredictor(cfg)
#
# from detectron2.utils.visualizer import ColorMode
# dataset_dicts = get_spine_dicts("/Users/xanderdavies/desktop/bkshlf/shelf/shelves/val")
#
# # %% test 5 images from val set
# for d in random.sample(dataset_dicts, 5):
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=shelf_metadata,
#                    scale=0.5,
#                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
#     )
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2.imshow("img", out.get_image()[:, :, ::-1])
#     cv2.waitKey()

# # %% evaluate performance
#
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader
# evaluator = COCOEvaluator("shelf_val", cfg, False, output_dir="/content/drive/My Drive/bkshlf/train")
#
# # %% evaluator = COCOEvaluator("shelf_val", cfg, False, output_dir="./output/")
# val_loader = build_detection_test_loader(cfg, "shelf_val")
# print(inference_on_dataset(trainer.model, val_loader, evaluator))
# # another equivalent way is to use trainer.test

"""# OCR time

## Create cropped images for each book
"""

# # %% Install dependencies
# !sudo apt install tesseract-ocr
# !pip install pytesseract

from imutils import rotate_bound
from PIL import Image, ImageDraw
import pytesseract
from pytesseract import Output
from matplotlib.image import imread
import scipy.misc
from detectron2.data import detection_utils
import re
# https://github.com/facebookresearch/detectron2/issues/984 helpful

# Need to add buffer around mask crop

def cropper(org_image_path, out_file_dir):

    filename = (org_image_path.split("/")[-1]).split(".")[0] # get file name
    img = detection_utils.read_image(org_image_path, format="BGR") # open image
    outputs = predictor(img)
    instances = outputs["instances"].to('cpu')
    pred_classes = instances.pred_classes
    classes = ["book_spine", "inc_spine", "no_text", "book_cover", "inc_cover"]
    labels = [classes[i] for i in pred_classes] # create labels for pred_classes
    mask_array = instances.pred_masks.numpy() # pred masks are now nd-numpy arrays

    num_instances = mask_array.shape[0] #6
    mask_array = np.moveaxis(mask_array, 0, -1)
    mask_array_instance = []
    img = imread(str(org_image_path))

    boxes = instances.pred_boxes
    if isinstance(boxes, detectron2.structures.boxes.Boxes):
        boxes = boxes.tensor.numpy()
    else:
        boxes = np.asarray(boxes)

    output = np.zeros_like(img)
    output_file_names = []

    for i in range(num_instances):
        if labels[i] == "book_spine":
            mask_array_instance.append(mask_array[:, :, i:(i+1)])
            output = np.where(mask_array_instance[i] == False, 0, img) # KEY LINE - if not mask array, then 255 (white), else copy from img
            # im = rotate_bound(output, 270) # rotate 270
            im = Image.fromarray(output)
            cropped_img = im.crop(boxes[i])
            # cropped_img = cropped_img.rotate(270)
            output_file_names.append(f"{out_file_dir}/{filename}_{i}.jpg")
            cropped_img.save(f"{out_file_dir}/{filename}_{i}.jpg")
            # im.save(f"{out_file_dir}/{filename}_{i}.jpg")

    return output_file_names

# import imutils
#
# def perspective_transform(image, corners):
#     def order_corner_points(corners):
#         # Separate corners into individual points
#         # Index 0 - top-right
#         #       1 - top-left
#         #       2 - bottom-left
#         #       3 - bottom-right
#         corners = [(corner[0][0], corner[0][1]) for corner in corners]
#         top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]
#         return (top_l, top_r, bottom_r, bottom_l)
#
#     # Order points in clockwise order
#     ordered_corners = order_corner_points(corners)
#     top_l, top_r, bottom_r, bottom_l = ordered_corners
#
#     # Determine width of new image which is the max distance between
#     # (bottom right and bottom left) or (top right and top left) x-coordinates
#     width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
#     width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
#     width = max(int(width_A), int(width_B))
#     print(width)
#
#     # Determine height of new image which is the max distance between
#     # (top right and bottom right) or (top left and bottom left) y-coordinates
#     height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
#     height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
#     height = max(int(height_A), int(height_B))
#
#     # Construct new points to obtain top-down view of image in
#     # top_r, top_l, bottom_l, bottom_r order
#     dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
#                     [0, height - 1]], dtype = "float32")
#
#     # Convert to Numpy format
#     ordered_corners = np.array(ordered_corners, dtype="float32")
#
#     # Find perspective transform matrix
#     matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)
#
#     # Return the transformed image
#     return cv2.warpPerspective(image, matrix, (width, height)), width
#
# image = cv2.imread(output_file_names[5])
# original = image.copy()
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#
# cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#
# width = 0
#
# for c in cnts:
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.015 * peri, True)
#     if len(approx) == 4:
#         cv2.drawContours(image,[c], 0, (36,255,12), 3)
#         transformed_n, width_n = perspective_transform(original, approx)
#         if width_n > width:
#             width = width_n
#             transformed = transformed_n
#
#
# rotated = imutils.rotate_bound(transformed, angle=-90)
# cv2_imshow(thresh)
# cv2_imshow(image)
# cv2_imshow(transformed)
# cv2_imshow(rotated)
#
# # !sudo apt install tesseract-ocr
# # !sudo add-apt-repository ppa:alex-p/tesseract-ocr
# # !sudo apt-get update
# # !sudo apt install tesseract-ocr
# # !tesseract -v # MUST BE V4
# # !pip install pillow
# # !pip install pytesseract
# # !pip install imutils
# # !tesseract --help-l
# # !tesseract --help-oem
# !tesseract --help-psm # 6 or 7? TRY CHANGING THIS

src = '/Users/xanderdavies/desktop/bkshlf/shelf/shelves/val/ideal.JPG'
out_folder = '/Users/xanderdavies/desktop/bkshlf/shelf/shelves/output_images'

# CROP BY PREDICTIONS
output_file_names = cropper(src, out_folder)

# """## Get text"""
#
# # import the necessary packages
# from imutils.object_detection import non_max_suppression
# from imutils import rotate_bound
# import numpy as np
# import pytesseract
# import argparse
# import cv2
#
# min_confidence = .5
# image_path = output_file_names[3]
# east_path = "/content/drive/My Drive/bkshlf/frozen_east_text_detection.pb"
# long_side = 672
# padding = 0.05 # PLAY WITH
#
# def decode_predictions(scores, geometry):
# 	# grab the number of rows and columns from the scores volume, then
# 	# initialize our set of bounding box rectangles and corresponding
# 	# confidence scores
# 	(numRows, numCols) = scores.shape[2:4]
# 	rects = []
# 	confidences = []
# 	# loop over the number of rows
# 	for y in range(0, numRows):
# 		# extract the scores (probabilities), followed by the
# 		# geometrical data used to derive potential bounding box
# 		# coordinates that surround text
# 		scoresData = scores[0, 0, y]
# 		xData0 = geometry[0, 0, y]
# 		xData1 = geometry[0, 1, y]
# 		xData2 = geometry[0, 2, y]
# 		xData3 = geometry[0, 3, y]
# 		anglesData = geometry[0, 4, y]
# 		# loop over the number of columns
# 		for x in range(0, numCols):
# 			# if our score does not have sufficient probability,
# 			# ignore it
# 			if scoresData[x] < min_confidence:
# 				continue
# 			# compute the offset factor as our resulting feature
# 			# maps will be 4x smaller than the input image
# 			(offsetX, offsetY) = (x * 4.0, y * 4.0)
# 			# extract the rotation angle for the prediction and
# 			# then compute the sin and cosine
# 			angle = anglesData[x]
# 			cos = np.cos(angle)
# 			sin = np.sin(angle)
# 			# use the geometry volume to derive the width and height
# 			# of the bounding box
# 			h = xData0[x] + xData2[x]
# 			w = xData1[x] + xData3[x]
# 			# compute both the starting and ending (x, y)-coordinates
# 			# for the text prediction bounding box
# 			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
# 			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
# 			startX = int(endX - w)
# 			startY = int(endY - h)
# 			# add the bounding box coordinates and probability score
# 			# to our respective lists
# 			rects.append((startX, startY, endX, endY))
# 			confidences.append(scoresData[x])
# 	# return a tuple of the bounding boxes and associated confidences
# 	return (rects, confidences)
#
# # load the input image and grab the image dimensions
# image = cv2.imread(image_path)
# image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) # ROTATE?
# orig = image.copy()
# (origH, origW) = image.shape[:2]
# # set the new width and height and then determine the ratio in change
# # for both the width and height
# if origH > origW:
#     short_side = int(((long_side/origH)*origW//32 + 1)*32)
#     (newW, newH) = (short_side, long_side)
# else:
#     short_side = int(((long_side/origW)*origH//32 + 1)*32)
#     (newW, newH) = (long_side, short_side)
#
# rW = origW / float(newW)
# rH = origH / float(newH)
# # resize the image and grab the new image dimensions
# image = cv2.resize(image, (newW, newH))
# (H, W) = image.shape[:2]
#
# # define the two output layer names for the EAST detector model that
# # we are interested in -- the first is the output probabilities and the
# # second can be used to derive the bounding box coordinates of text
# layerNames = [
# 	"feature_fusion/Conv_7/Sigmoid",
# 	"feature_fusion/concat_3"]
# # load the pre-trained EAST text detector
# print("[INFO] loading EAST text detector...")
# net = cv2.dnn.readNet(east_path)
#
# # construct a blob from the image and then perform a forward pass of
# # the model to obtain the two output layer sets
# blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
# 	(123.68, 116.78, 103.94), swapRB=True, crop=False)
# net.setInput(blob)
# (scores, geometry) = net.forward(layerNames)
# # decode the predictions, then  apply non-maxima suppression to
# # suppress weak, overlapping bounding boxes
# (rects, confidences) = decode_predictions(scores, geometry)
# if rects == []:
#   print("failed to locate text")
# cv2_imshow(image)
#
# boxes = non_max_suppression(np.array(rects), probs=confidences)
#
# # initialize the list of results
# results = []
# # loop over the bounding boxes
# for (startX, startY, endX, endY) in boxes:
# 	# scale the bounding box coordinates based on the respective
# 	# ratios
# 	startX = int(startX * rW)
# 	startY = int(startY * rH)
# 	endX = int(endX * rW)
# 	endY = int(endY * rH)
# 	# in order to obtain a better OCR of the text we can potentially
# 	# apply a bit of padding surrounding the bounding box -- here we
# 	# are computing the deltas in both the x and y directions
# 	dX = int((endX - startX) * padding)
# 	dY = int((endY - startY) * padding)
# 	# apply padding to each side of the bounding box, respectively
# 	startX = max(0, startX - dX)
# 	startY = max(0, startY - dY)
# 	endX = min(origW, endX + (dX * 2))
# 	endY = min(origH, endY + (dY * 2))
# 	# extract the actual padded ROI
# 	roi = orig[startY:endY, startX:endX]
#
#   # in order to apply Tesseract v4 to OCR text we must supply
# 	# (1) a language, (2) an OEM flag of 4, indicating that the we
# 	# wish to use the LSTM neural net model for OCR, and finally
# 	# (3) an OEM value, in this case, 7 which implies that we are
# 	# treating the ROI as a single line of text
# 	config = ("-l eng --oem 1 --psm 8")
# 	text = pytesseract.image_to_string(roi, config=config)
# 	# add the bounding box coordinates and OCR'd text to the list
# 	# of results
# 	results.append(((startX, startY, endX, endY), text))
#
# # sort the results bounding box coordinates from top to bottom
# results = sorted(results, key=lambda r:r[0][1])
#
# book_text = []
# for word in results:
#   book_text.append(word[1])
# book_text = ' '.join(book_text)
# print(book_text)
#
# # loop over the results
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
#   cv2_imshow(output)
#
# """# Google Books API"""
#
# import requests
# import json
#
# APIKEY = "AIzaSyCQCfV4eIoFOdWkXClJtPJYqWMU0Gds9RE"
# sample_text = "Fodors aii GRAND CANYON 2016"
#
# class Book:
#   def __init__(self, title, authors, id, description):
#     self.title = title
#     self.authors = authors
#     self.id = id
#     self.description = description
#
# STRING = sample_text
# STRING = '+'.join(STRING.split())
# resp = requests.get(f"https://www.googleapis.com/books/v1/volumes?q={STRING}&key={APIKEY}")
# j_resp = resp.json()
# print(j_resp)
# print(j_resp["items"][0]["volumeInfo"])
#
# title = j_resp["items"][0]["volumeInfo"]["title"]
# authors = j_resp["items"][0]["volumeInfo"]["authors"]
# id = j_resp["items"][0]["id"]
# description = j_resp["items"][0]["volumeInfo"]["description"]
#
# my_book = Book(title, authors, id, description)
# print(f"Book Title: {my_book.title}")
# print(f"Book Description: {my_book.description}")
#
#
#
# """# *Old Code*"""
#
# # import the necessary packages
# import numpy as np
# import cv2
# def order_points(pts):
# 	# initialzie a list of coordinates that will be ordered
# 	# such that the first entry in the list is the top-left,
# 	# the second entry is the top-right, the third is the
# 	# bottom-right, and the fourth is the bottom-left
# 	rect = np.zeros((4, 2), dtype = "float32")
# 	# the top-left point will have the smallest sum, whereas
# 	# the bottom-right point will have the largest sum
# 	s = pts.sum(axis = 1)
# 	rect[0] = pts[np.argmin(s)]
# 	rect[2] = pts[np.argmax(s)]
# 	# now, compute the difference between the points, the
# 	# top-right point will have the smallest difference,
# 	# whereas the bottom-left will have the largest difference
# 	diff = np.diff(pts, axis = 1)
# 	rect[1] = pts[np.argmin(diff)]
# 	rect[3] = pts[np.argmax(diff)]
# 	# return the ordered coordinates
# 	return rect
#
# # HI HI HI
# # HIIIII
#
# from imutils.object_detection import non_max_suppression
# import numpy as np
# import time
# import cv2
#
# src = '/content/drive/My Drive/bkshlf/val/IMG_1350.JPG'
# out_folder = '/content/drive/My Drive/bkshlf/output_images'
# given_width = 320
# given_height = 320
# min_confidence = .5
# east_path = "/content/drive/My Drive/bkshlf/frozen_east_text_detection.pb"
#
# # # Display starting image
# # img = cv2.imread(src)
# # cv2_imshow(img)
#
# # CROP BY PREDICTIONS
# output_file_names = cropper(src, out_folder)
#
# # import the necessary packages
# from imutils.object_detection import non_max_suppression
# import numpy as np
# import time
# import cv2
#
# book = output_file_names[0]
# print(book)
#
# books = []
# # ROTATE + RESIZE + READ BOOKS (for fun)
# for book in output_file_names:
#     book = cv2.imread(book)
#     book = cv2.rotate(book, cv2.ROTATE_90_COUNTERCLOCKWISE)
#     print(f"Approximate leter height: {book.shape[0]/2}")
#     scale_percent = 90/(book.shape[0]/2) * 100 # Comment out OLD: 100000/book.shape[0]
#     print(f"Scaled by {scale_percent}%")
#     width = int(book.shape[1] * scale_percent / 100)
#     height = int(book.shape[0] * scale_percent / 100)
#     print(height)
#     dsize = (width, height)
#     book = cv2.resize(book, dsize)
#     cv2_imshow(book)
#     book = cv2.cvtColor(book, cv2.COLOR_BGR2RGB)
#     book_text = pytesseract.image_to_string(book)
#     book_text = re.sub(r"(?:^| )\w(?:$| )", " ", book_text).strip() # Remove single characters
#     book_text = re.sub(r"\W+|_", " ", book_text) # Remove special characters
#     print(f"text extracted: {book_text}")
#     print(f"full data: {pytesseract.image_to_data(book)}")
#     books.append(book_text)
#
# print(books)
#
# # import the necessary packages
# import numpy as np
# import cv2
# def order_points(pts):
# 	# initialzie a list of coordinates that will be ordered
# 	# such that the first entry in the list is the top-left,
# 	# the second entry is the top-right, the third is the
# 	# bottom-right, and the fourth is the bottom-left
# 	rect = np.zeros((4, 2), dtype = "float32")
# 	# the top-left point will have the smallest sum, whereas
# 	# the bottom-right point will have the largest sum
# 	s = pts.sum(axis = 1)
# 	rect[0] = pts[np.argmin(s)]
# 	rect[2] = pts[np.argmax(s)]
# 	# now, compute the difference between the points, the
# 	# top-right point will have the smallest difference,
# 	# whereas the bottom-left will have the largest difference
# 	diff = np.diff(pts, axis = 1)
# 	rect[1] = pts[np.argmin(diff)]
# 	rect[3] = pts[np.argmax(diff)]
# 	# return the ordered coordinates
# 	return rect
#
# # for i, d in enumerate(random.sample(dataset_dicts, 2)):
# #     # read the image with cv2
# #     img = cv2.imread(d["file_name"])
# #     visualizer = Visualizer(img[:, :, ::-1], metadata=building_metadata, scale=0.5)
# #     vis = visualizer.draw_dataset_dict(d)
# #     cv2_imshow(vis.get_image()[:, :, ::-1])
# #     # if you want to save the files, uncomment the line below, but keep in mind that
# #     # the folder inputs has to be created first
# #     # plt.savefig(f"./inputs/input_{i}.jpg")
#
# # cfg = get_cfg()
# # # you can choose alternative models as backbone here
# # cfg.merge_from_file(model_zoo.get_config_file(
# #     "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
# # ))
#
# # cfg.DATASETS.TRAIN = ("shelf_train",)
# # cfg.DATASETS.TEST = ()
# # cfg.DATALOADER.NUM_WORKERS = 0
# # # if you changed the model above, you need to adapt the following line as well
# # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
# #     "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
# # )  # Let training initialize from model zoo
# # cfg.SOLVER.IMS_PER_BATCH = 2
# # cfg.SOLVER.BASE_LR = 0.00015  # pick a good LR, 0.00025 seems a good start
# # cfg.SOLVER.MAX_ITER = (
# #     1000  # 1000 iterations is a good start, for better accuracy increase this value
# # )
# # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
# #     512  # (default: 512), select smaller if faster training is needed
# # )
# # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # for the two classes window and building
#
# # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# # trainer = DefaultTrainer(cfg)
# # trainer.resume_or_load(resume=False)
# # trainer.train()
#
# # !mkdir predictions
# # !mkdir output_images
#
# # !nvidia-smi
#
# # !python -m detectron2.utils.collect_env
#
# # def get_spine_dicts(img_dir):
# #     """This function loads the JSON file created with the annotator and converts it to
# #     the detectron2 metadata specifications.
# #     """
# #     # load the JSON file
# #     json_file = os.path.join(img_dir, "via_region_data.json")
# #     with open(json_file) as f:
# #         imgs_anns = json.load(f)
#
# #     dataset_dicts = []
# #     # loop through the entries in the JSON file
# #     for idx, v in enumerate(imgs_anns.values()):
# #         record = {}
# #         # add file_name, image_id, height and width information to the records
# #         filename = os.path.join(img_dir, v["filename"])
# #         print(filename)
# #         height, width = cv2.imread(filename).shape[:2]
#
# #         record["file_name"] = filename
# #         record["image_id"] = idx
# #         record["height"] = height
# #         record["width"] = width
#
# #         annos = v["regions"]
#
# #         objs = []
# #         # one image can have multiple annotations, therefore this loop is needed
# #         for annotation in annos:
# #             # reformat the polygon information to fit the specifications
# #             anno = annotation["shape_attributes"]
# #             px = anno["all_points_x"]
# #             py = anno["all_points_y"]
# #             poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
# #             poly = [p for x in poly for p in x]
#
# #             region_attributes = annotation["region_attributes"]["class"]
#
# #             # specify the category_id to match with the class.
#
# #             if "book_spine" in region_attributes:
# #                 category_id = 0
# #             elif "inc_spine" in region_attributes:
# #                 category_id = 1
# #             elif "no_text" in region_attributes:
# #                 category_id = 2
# #             elif "book_cover" in region_attributes:
# #                 category_id = 3
# #             elif "inc_cover" in region_attributes:
# #                 category_id = 4
#
# #             obj = {
# #                 "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
# #                 "bbox_mode": BoxMode.XYXY_ABS,
# #                 "segmentation": [poly],
# #                 "category_id": category_id,
# #                 "iscrowd": 0,
# #             }
# #             objs.append(obj)
# #         record["annotations"] = objs
# #         dataset_dicts.append(record)
#
# #     return dataset_dicts
#
# # # the data has to be registered within detectron2, once for the train and once for
# # # the val data
# # for d in ["train", "val"]:
# #     DatasetCatalog.register(
# #         "shelf_" + d,
# #         lambda d=d: get_spine_dicts("/content/drive/My Drive/bkshlf/" + d),
# #     )
#
# # building_metadata = MetadataCatalog.get("shelf_train")
#
# # dataset_dicts = get_spine_dicts("/content/drive/My Drive/bkshlf/train")
