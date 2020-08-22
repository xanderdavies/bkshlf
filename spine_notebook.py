# BKSHLF end-to-end workflow

# FOR MAX, CURRENT TODO/ISSUES:
# 1. Try goodreads DONE
# 2. Handle autobrightness
# 3. Make sure at least one word detected is in author or publisher name DONE
# 4. Books where title is spelled out one letter at a time :/
# 5. Has to be horizontal image rn?
# 6. Teach the ocr new fonts?
# 7. Don't always do all four reads... do the first two and see if necessary


from ocr import cropper, image_reader
from isbn_api import text_to_book
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
user = 'M'

# %% define paths + initialize book_list
if user == 'X':
    path_to_image = "/Users/xanderdavies/Desktop/bkshlf/shelf/shelves/val/ideal.jpg"
    path_to_out = "/Users/xanderdavies/Desktop/bkshlf/shelf/shelves/output_images"
    path_to_weights = "/Users/xanderdavies/Desktop/bkshlf/shelf/shelves/saved_models/model_final.pth"

elif user == 'M':
    path_to_image = "/Users/maxnadeau/Documents/ExtraProjects/bookshelf/new_bs/shelves/val/ideal.JPG"
    path_to_out = "/Users/maxnadeau/Documents/ExtraProjects/bookshelf/new_bs/shelves/output_images"
    path_to_weights = "/Users/maxnadeau/Documents/ExtraProjects/bookshelf/new_bs/shelves/saved_models/model_final.pth"

book_list = []

# %% define model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = path_to_weights
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  # guess
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.DATASETS.TEST = ()
predictor = DefaultPredictor(cfg)

# %% run cropper
#output_file_names = cropper(path_to_image, path_to_out, predictor)
image_reader(path_to_out + "/ideal_0.jpg")
#output_file_names = ["ideal_0", "ideal_1", "ideal_2"]

# %% run image_reader and text_to_book
for i, file in enumerate(output_file_names):
    read_image = image_reader(file)
    if read_image != ("", ""):
        book = text_to_book(read_image)
        book_list.append(book)

# %% output
for i, book in enumerate(book_list):
    if book == None:
        print(f"{i+1}. NA")
    else:
        try:
            print(f"{i+1}. {book.title} by {book.authors[0]}")
        except IndexError:
            print(f"{i+1}. {book.title} by {book.authors}")
###############################
# %% possible pip installs ((colab has CUDA 10.1 + torch 1.6)):
# HELPFUL !conda install -c pytorch torchvision cudatoolkit pytorch
# !sudo -H pip3 install --ignore-installed PyYAML
# !pip install --user pyyaml==5.1 pycocotools>=2.0.1
# !pip install opencv-python
# !sudo pip install torch --upgrade
# !pip install torch==1.6.0
# !pip install torchvision==0.7.0

# %% OLD CODE BELOW


# I don't think any of these cfg settings are needed for testing...
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.DATASETS.TRAIN = ("shelf_train",)
# cfg.DATASETS.TEST = ()
# cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
# cfg.SOLVER.IMS_PER_BATCH = 2
# cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
# cfg.SOLVER.MAX_ITER = 1000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # default: 512, was 128
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only one class rn

# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") # NEED TO UNCOMMENT TO USE FRESHLY TRAINED MODEL
#
# # if loading:
# # Dowloaded from /output in this collab
# cfg.MODEL.WEIGHTS = '/content/drive/My Drive/bkshlf/saved_models/model_final.pth'
# # GUESS — custom testing threshold for this model
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
# cfg.DATASETS.TEST = ("shelf_val", )
#
# # %% codecell
# # the data has to be registered within detectron2
# for d in ["train", "val"]:
#     DatasetCatalog.register(
#         "shelf_" + d, lambda d=d: get_spine_dicts("./shelves/" + d))
#     MetadataCatalog.get(
#         "shelf_" + d).set(thing_classes=["book_spine", "not_spine"])
#     # shelf_metadata = MetadataCatalog.get("shelf_train").set(thing_classes=["book_spine", "not_spine"])
#
# shelf_metadata = MetadataCatalog.get("shelf_train")
#
#
# # %% codecell
# # display a few images to confirm they were imported currently
# dataset_dicts = get_spine_dicts("./shelves/train")
#
# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(
#         img[:, :, ::-1], metadata=shelf_metadata, scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     cv2_imshow(out.get_image()[:, :, ::-1])
#
# # %% codecell
# # test 5 images from val set
# dataset_dicts = get_spine_dicts("./shelves/val")
#
# for d in random.sample(dataset_dicts, 5):
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=shelf_metadata,
#                    scale=0.5,
#                    # remove the colors of unsegmented pixels. This option is only available for segmentation models
#                    instance_mode=ColorMode.IMAGE_BW
#                    )
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2_imshow(out.get_image()[:, :, ::-1])
