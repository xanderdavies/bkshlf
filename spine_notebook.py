# Need train + val data @ "/content/drive/My Drive/bkshlf/(train / val)"


# Install detectron2

# %% possible pip installs ((colab has CUDA 10.1 + torch 1.6)):
# !sudo -H pip3 install --ignore-installed PyYAML
# !pip install --user pyyaml==5.1 pycocotools>=2.0.1
# !pip install opencv-python
# !sudo pip install torch --upgrade
# !pip install torch==1.6.0
# !pip install torchvision==0.7.0

# %%
from prepare_data import get_spine_dicts
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import random
import cv2

cfg = get_cfg()

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

# if loading:
cfg.MODEL.WEIGHTS = '/content/drive/My Drive/bkshlf/saved_models/model_final.pth' # Dowloaded from /output in this collab

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  # GUESS — custom testing threshold for this model
cfg.DATASETS.TEST = ("shelf_val", )

# %% codecell
# the data has to be registered within detectron2
for d in ["train", "val"]:
    DatasetCatalog.register("shelf_" + d, lambda d=d: get_spine_dicts("./shelves/" + d))
    MetadataCatalog.get("shelf_" + d).set(thing_classes=["book_spine", "not_spine"])
    # shelf_metadata = MetadataCatalog.get("shelf_train").set(thing_classes=["book_spine", "not_spine"])

shelf_metadata = MetadataCatalog.get("shelf_train")


# %% codecell
# display a few images to confirm they were imported currently
dataset_dicts = get_spine_dicts("./shelves/train")

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=shelf_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2_imshow(out.get_image()[:, :, ::-1])

# %% codecell
# test 5 images from val set
dataset_dicts = get_spine_dicts("./shelves/val")

for d in random.sample(dataset_dicts, 5):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=shelf_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])
