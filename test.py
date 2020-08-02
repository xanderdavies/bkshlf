# %% import torch and confirm version
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.6")
# !conda install -c conda-forge detectron2
# !pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html

# %% basic setup, common libraries
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger() # Setup detectron2 logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColoMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import pickle
from pathlib import Path
from tqdm import tqdm
from google.colab.patches import cv2_imshow

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import random

from prepare_data import get_spine_dicts


# # Test the model

# %% codecell


# %% codecell

def run_model (cfg) :
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
