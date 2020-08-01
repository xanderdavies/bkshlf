# # *Old Code*
# %% codecell
# import the necessary packages
import numpy as np
import cv2



def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

# HI HI HI
# HIIIII
# %% codecell
from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2

src = '/content/drive/My Drive/bkshlf/val/IMG_1350.JPG'
out_folder = '/content/drive/My Drive/bkshlf/output_images'
given_width = 320
given_height = 320
min_confidence = .5
east_path = "/content/drive/My Drive/bkshlf/frozen_east_text_detection.pb"

# # Display starting image
# img = cv2.imread(src)
# cv2_imshow(img)

# CROP BY PREDICTIONS
output_file_names = cropper(src, out_folder)
# %% codecell
# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2

book = output_file_names[0]
print(book)

# %% codecell
books = []
# ROTATE + RESIZE + READ BOOKS (for fun)
for book in output_file_names:
    book = cv2.imread(book)
    book = cv2.rotate(book, cv2.ROTATE_90_COUNTERCLOCKWISE)
    print(f"Approximate leter height: {book.shape[0]/2}")
    scale_percent = 90/(book.shape[0]/2) * 100 # Comment out OLD: 100000/book.shape[0]
    print(f"Scaled by {scale_percent}%")
    width = int(book.shape[1] * scale_percent / 100)
    height = int(book.shape[0] * scale_percent / 100)
    print(height)
    dsize = (width, height)
    book = cv2.resize(book, dsize)
    cv2_imshow(book)
    book = cv2.cvtColor(book, cv2.COLOR_BGR2RGB)
    book_text = pytesseract.image_to_string(book)
    book_text = re.sub(r"(?:^| )\w(?:$| )", " ", book_text).strip() # Remove single characters
    book_text = re.sub(r"\W+|_", " ", book_text) # Remove special characters
    print(f"text extracted: {book_text}")
    print(f"full data: {pytesseract.image_to_data(book)}")
    books.append(book_text)

print(books)
# %% codecell
# import the necessary packages
import numpy as np
import cv2
def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect
# %% codecell
# for i, d in enumerate(random.sample(dataset_dicts, 2)):
#     # read the image with cv2
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=building_metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
#     cv2_imshow(vis.get_image()[:, :, ::-1])
#     # if you want to save the files, uncomment the line below, but keep in mind that
#     # the folder inputs has to be created first
#     # plt.savefig(f"./inputs/input_{i}.jpg")
# %% codecell
# cfg = get_cfg()
# # you can choose alternative models as backbone here
# cfg.merge_from_file(model_zoo.get_config_file(
#     "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
# ))

# cfg.DATASETS.TRAIN = ("shelf_train",)
# cfg.DATASETS.TEST = ()
# cfg.DATALOADER.NUM_WORKERS = 0
# # if you changed the model above, you need to adapt the following line as well
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
#     "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
# )  # Let training initialize from model zoo
# cfg.SOLVER.IMS_PER_BATCH = 2
# cfg.SOLVER.BASE_LR = 0.00015  # pick a good LR, 0.00025 seems a good start
# cfg.SOLVER.MAX_ITER = (
#     1000  # 1000 iterations is a good start, for better accuracy increase this value
# )
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
#     512  # (default: 512), select smaller if faster training is needed
# )
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # for the two classes window and building
# %% codecell
# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()
# %% codecell
# !mkdir predictions
# !mkdir output_images
# %% codecell
# !nvidia-smi
# %% codecell
# !python -m detectron2.utils.collect_env
# %% codecell
# def get_spine_dicts(img_dir):
#     """This function loads the JSON file created with the annotator and converts it to
#     the detectron2 metadata specifications.
#     """
#     # load the JSON file
#     json_file = os.path.join(img_dir, "via_region_data.json")
#     with open(json_file) as f:
#         imgs_anns = json.load(f)

#     dataset_dicts = []
#     # loop through the entries in the JSON file
#     for idx, v in enumerate(imgs_anns.values()):
#         record = {}
#         # add file_name, image_id, height and width information to the records
#         filename = os.path.join(img_dir, v["filename"])
#         print(filename)
#         height, width = cv2.imread(filename).shape[:2]

#         record["file_name"] = filename
#         record["image_id"] = idx
#         record["height"] = height
#         record["width"] = width

#         annos = v["regions"]

#         objs = []
#         # one image can have multiple annotations, therefore this loop is needed
#         for annotation in annos:
#             # reformat the polygon information to fit the specifications
#             anno = annotation["shape_attributes"]
#             px = anno["all_points_x"]
#             py = anno["all_points_y"]
#             poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
#             poly = [p for x in poly for p in x]

#             region_attributes = annotation["region_attributes"]["class"]

#             # specify the category_id to match with the class.

#             if "book_spine" in region_attributes:
#                 category_id = 0
#             elif "inc_spine" in region_attributes:
#                 category_id = 1
#             elif "no_text" in region_attributes:
#                 category_id = 2
#             elif "book_cover" in region_attributes:
#                 category_id = 3
#             elif "inc_cover" in region_attributes:
#                 category_id = 4

#             obj = {
#                 "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
#                 "bbox_mode": BoxMode.XYXY_ABS,
#                 "segmentation": [poly],
#                 "category_id": category_id,
#                 "iscrowd": 0,
#             }
#             objs.append(obj)
#         record["annotations"] = objs
#         dataset_dicts.append(record)

#     return dataset_dicts
# %% codecell
# # the data has to be registered within detectron2, once for the train and once for
# # the val data
# for d in ["train", "val"]:
#     DatasetCatalog.register(
#         "shelf_" + d,
#         lambda d=d: get_spine_dicts("/content/drive/My Drive/bkshlf/" + d),
#     )

# building_metadata = MetadataCatalog.get("shelf_train")

# dataset_dicts = get_spine_dicts("/content/drive/My Drive/bkshlf/train")

# GOES TO OLD codecell
# %% codecell
import imutils

def perspective_transform(image, corners):
    def order_corner_points(corners):
        # Separate corners into individual points
        # Index 0 - top-right
        #       1 - top-left
        #       2 - bottom-left
        #       3 - bottom-right
        corners = [(corner[0][0], corner[0][1]) for corner in corners]
        top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]
        return (top_l, top_r, bottom_r, bottom_l)

    # Order points in clockwise order
    ordered_corners = order_corner_points(corners)
    top_l, top_r, bottom_r, bottom_l = ordered_corners

    # Determine width of new image which is the max distance between
    # (bottom right and bottom left) or (top right and top left) x-coordinates
    width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
    width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))
    print(width)

    # Determine height of new image which is the max distance between
    # (top right and bottom right) or (top left and bottom left) y-coordinates
    height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
    height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
    height = max(int(height_A), int(height_B))

    # Construct new points to obtain top-down view of image in
    # top_r, top_l, bottom_l, bottom_r order
    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                    [0, height - 1]], dtype = "float32")

    # Convert to Numpy format
    ordered_corners = np.array(ordered_corners, dtype="float32")

    # Find perspective transform matrix
    matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

    # Return the transformed image
    return cv2.warpPerspective(image, matrix, (width, height)), width

image = cv2.imread(output_file_names[5])
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

width = 0

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.015 * peri, True)
    if len(approx) == 4:
        cv2.drawContours(image,[c], 0, (36,255,12), 3)
        transformed_n, width_n = perspective_transform(original, approx)
        if width_n > width:
            width = width_n
            transformed = transformed_n


rotated = imutils.rotate_bound(transformed, angle=-90)
cv2_imshow(thresh)
cv2_imshow(image)
cv2_imshow(transformed)
cv2_imshow(rotated)
