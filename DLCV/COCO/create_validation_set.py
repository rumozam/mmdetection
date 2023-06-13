import json
import numpy as np

UNSEEN_CLASSES = [{"supercategory": "vehicle", "id": 5, "name": "airplane"},
{"supercategory": "vehicle", "id": 7, "name": "train"},
{"supercategory": "outdoor", "id": 14, "name": "parking meter"},
{"supercategory": "animal", "id": 17, "name": "cat"},
{"supercategory": "animal", "id": 23, "name": "bear"},
{"supercategory": "accessory", "id": 33, "name": "suitcase"},
{"supercategory": "sports", "id": 34, "name": "frisbee"},
{"supercategory": "sports", "id": 36, "name": "snowboard"},
{"supercategory": "kitchen", "id": 48, "name": "fork"},
{"supercategory": "food", "id": 54, "name": "sandwich"},
{"supercategory": "food", "id": 58, "name": "hot dog"},
{"supercategory": "furniture", "id": 70, "name": "toilet"},
{"supercategory": "electronic", "id": 74, "name": "mouse"},
{"supercategory": "appliance", "id": 80, "name": "toaster"},
{"supercategory": "indoor", "id": 89, "name": "hair drier"}]

KEYS = ["info", "images", "licenses", "annotations", "categories"]
KEYS_INFO = ["description", "url", "version", "year", "contributor", "date_created"]
KEYS_IMAGES = ["license", "file_name", "coco_url", "height", "width", "date_captured", "flickr_url", "id"]
KEYS_LICENSES = ["url", "id", "name"]
KEYS_ANNOTATIONS = ["segmentation", "area", "iscrowd", "image_id", "bbox", "category_id", "id"]
KEYS_CATEGORIES = ["supercategory", "id", "name"]

DIR_BASE = "../../data/coco/annotations/"


def create_validation_set():
    unseen_ids = []
    for cls in UNSEEN_CLASSES:
        unseen_ids.append(cls["id"])

    with open(DIR_BASE + "instances_train2014.json") as train_file:
        data = json.load(train_file)

        image_ids_to_annotation = dict()
        for ann in data["annotations"]:
            if ann["category_id"] in unseen_ids:
                if ann["image_id"] in image_ids_to_annotation:
                    image_ids_to_annotation[ann["image_id"]].append(ann)
                else:
                    image_ids_to_annotation[ann["image_id"]] = [ann]

    dataset_unseen_annotations_from_training_set = {"info" : data["info"], "images" : [], "licenses" : data["licenses"],
                                "annotations" : [], "categories" : data["categories"]}

    img_ids = list(image_ids_to_annotation.keys())
    for img in img_ids:
        dataset_unseen_annotations_from_training_set["images"].append(img)
    print(f"{len(img_ids)} Images added")

    for img in img_ids:
        for ann in image_ids_to_annotation[img]:
            dataset_unseen_annotations_from_training_set["annotations"].append(ann)

    print(f'{len(dataset_unseen_annotations_from_training_set["annotations"])} Annotations added')


    with open(f"/fastdata/vires01/coco/annotations/validation.json", "w") as outfile:
        json.dump(dataset_unseen_annotations_from_training_set, outfile, indent=2)
        print("JSON-file saved")


if __name__ == "__main__":
    create_validation_set()
