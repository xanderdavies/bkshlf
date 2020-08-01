# Need train + val data @ "/content/drive/My Drive/bkshlf/(train / val)"


# Install detectron2

# %% possible pip installs ((colab has CUDA 10.1 + torch 1.6)):
# !pip install pyyaml==5.1 pycocotools>=2.0.1
# !pip install detectron2==0.2.1 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
# !pip install opencv-python
# !sudo pip install torch --upgrade
# !pip install torch==1.6.0

# %% import torch and confirm version
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.6")

# %% basic setup, common libraries
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger() # Setup detectron2 logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColoMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
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
from google.colab.patches import cv2_imshow


# # Prepare the dataset
# %% codecell
# if we export dataset as COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

def get_spine_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        print(filename)

        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for annotation in annos:
            anno = annotation["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            # specify the category_id to match with the class.
            region_attributes = annotation["region_attributes"]["class"]

            if "book_spine" in region_attributes:
                category_id = 0
            elif "inc_spine" in region_attributes:
                category_id = 0
            elif "no_text" in region_attributes:
                category_id = 0
            elif "book_cover" in region_attributes:
                category_id = 1
            elif "inc_cover" in region_attributes:
                category_id = 1

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": category_id,
            }

            if obj["category_id"] == 0: # ONLY LEARN SPINES
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

# the data has to be registered within detectron2
for d in ["train", "val"]:
    DatasetCatalog.register("shelf_" + d, lambda d=d: get_spine_dicts("/content/drive/My Drive/bkshlf/" + d))
    MetadataCatalog.get("shelf_" + d).set(thing_classes=["book_spine", "not_spine"])
    # shelf_metadata = MetadataCatalog.get("shelf_train").set(thing_classes=["book_spine", "not_spine"])

shelf_metadata = MetadataCatalog.get("shelf_train")
# %% codecell
dataset_dicts = get_spine_dicts("/content/drive/My Drive/bkshlf/train")

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=shelf_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2_imshow(out.get_image()[:, :, ::-1])


# # Test the model
# %% codecell
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") # NEED TO UNCOMMENT TO USE FRESHLY TRAINED MODEL

# if loading:
cfg.MODEL.WEIGHTS = '/content/drive/My Drive/bkshlf/saved_models/model_final.pth' # Dowloaded from /output in this collab

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  # GUESS — custom testing threshold for this model
cfg.DATASETS.TEST = ("shelf_val", )
predictor = DefaultPredictor(cfg)
# %% codecell

dataset_dicts = get_spine_dicts("/content/drive/My Drive/bkshlf/val")
# %% codecell
# test 5 images from val set
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
# %% codecell
# evaluate performance


evaluator = COCOEvaluator("shelf_val", cfg, False, output_dir="/content/drive/My Drive/bkshlf/train")

# evaluator = COCOEvaluator("shelf_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "shelf_val")
print(inference_on_dataset(trainer.model, val_loader, evaluator))
# another equivalent way is to use trainer.test
