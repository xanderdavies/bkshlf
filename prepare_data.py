# Prepare the dataset

# %% imports
import os
import cv2
import json
import numpy as np
import detectron2
from detectron2.structures import BoxMode

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
