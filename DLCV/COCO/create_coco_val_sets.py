import json
import numpy as np

SEEN_CLASSES = [{"supercategory": "person", "id": 1, "name": "person"},
{"supercategory": "vehicle", "id": 2, "name": "bicycle"},
{"supercategory": "vehicle", "id": 3, "name": "car"},
{"supercategory": "vehicle", "id": 4, "name": "motorcycle"},
{"supercategory": "vehicle", "id": 6, "name": "bus"},
{"supercategory": "vehicle", "id": 8, "name": "truck"},
{"supercategory": "vehicle", "id": 9, "name": "boat"},
{"supercategory": "outdoor", "id": 10, "name": "traffic light"},
{"supercategory": "outdoor", "id": 11, "name": "fire hydrant"},
{"supercategory": "outdoor", "id": 13, "name": "stop sign"},
{"supercategory": "outdoor", "id": 15, "name": "bench"},
{"supercategory": "animal", "id": 16, "name": "bird"},
{"supercategory": "animal", "id": 18, "name": "dog"},
{"supercategory": "animal", "id": 19, "name": "horse"},
{"supercategory": "animal", "id": 20, "name": "sheep"},
{"supercategory": "animal", "id": 21, "name": "cow"},
{"supercategory": "animal", "id": 22, "name": "elephant"},
{"supercategory": "animal", "id": 24, "name": "zebra"},
{"supercategory": "animal", "id": 25, "name": "giraffe"},
{"supercategory": "accessory", "id": 27, "name": "backpack"},
{"supercategory": "accessory", "id": 28, "name": "umbrella"},
{"supercategory": "accessory", "id": 31, "name": "handbag"},
{"supercategory": "accessory", "id": 32, "name": "tie"},
{"supercategory": "sports", "id": 35, "name": "skis"},
{"supercategory": "sports", "id": 37, "name": "sports ball"},
{"supercategory": "sports", "id": 38, "name": "kite"},
{"supercategory": "sports", "id": 39, "name": "baseball bat"},
{"supercategory": "sports", "id": 40, "name": "baseball glove"},
{"supercategory": "sports", "id": 41, "name": "skateboard"},
{"supercategory": "sports", "id": 42, "name": "surfboard"},
{"supercategory": "sports", "id": 43, "name": "tennis racket"},
{"supercategory": "kitchen", "id": 44, "name": "bottle"},
{"supercategory": "kitchen", "id": 46, "name": "wine glass"},
{"supercategory": "kitchen", "id": 47, "name": "cup"},
{"supercategory": "kitchen", "id": 49, "name": "knife"},
{"supercategory": "kitchen", "id": 50, "name": "spoon"},
{"supercategory": "kitchen", "id": 51, "name": "bowl"},
{"supercategory": "food", "id": 52, "name": "banana"},
{"supercategory": "food", "id": 53, "name": "apple"},
{"supercategory": "food", "id": 55, "name": "orange"},
{"supercategory": "food", "id": 56, "name": "broccoli"},
{"supercategory": "food", "id": 57, "name": "carrot"},
{"supercategory": "food", "id": 59, "name": "pizza"},
{"supercategory": "food", "id": 60, "name": "donut"},
{"supercategory": "food", "id": 61, "name": "cake"},
{"supercategory": "furniture", "id": 62, "name": "chair"},
{"supercategory": "furniture", "id": 63, "name": "couch"},
{"supercategory": "furniture", "id": 64, "name": "potted plant"},
{"supercategory": "furniture", "id": 65, "name": "bed"},
{"supercategory": "furniture", "id": 67, "name": "dining table"},
{"supercategory": "electronic", "id": 72, "name": "tv"},
{"supercategory": "electronic", "id": 73, "name": "laptop"},
{"supercategory": "electronic", "id": 75, "name": "remote"},
{"supercategory": "electronic", "id": 76, "name": "keyboard"},
{"supercategory": "electronic", "id": 77, "name": "cell phone"},
{"supercategory": "appliance", "id": 78, "name": "microwave"},
{"supercategory": "appliance", "id": 79, "name": "oven"},
{"supercategory": "appliance", "id": 81, "name": "sink"},
{"supercategory": "appliance", "id": 82, "name": "refrigerator"},
{"supercategory": "indoor", "id": 84, "name": "book"},
{"supercategory": "indoor", "id": 85, "name": "clock"},
{"supercategory": "indoor", "id": 86, "name": "vase"},
{"supercategory": "indoor", "id": 87, "name": "scissors"},
{"supercategory": "indoor", "id": 88, "name": "teddy bear"},
{"supercategory": "indoor", "id": 90, "name": "toothbrush"}]
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


def create_zsd_and_gzsd_val_sets():
    unseen_ids = []
    for cls in UNSEEN_CLASSES:
        unseen_ids.append(cls["id"])

    with open(DIR_BASE + "instances_val2014_unseen_65_15.json") as val_file:
        data = json.load(val_file)

        imgs = []
        for ann in data["annotations"]:
            if ann["image_id"] not in imgs:
                imgs.append(ann["image_id"])


    with open(DIR_BASE + "instances_val2014.json") as val_file:
        data = json.load(val_file)

        dataset_unseen = {"info" : data["info"], "images" : [], "licenses" : data["licenses"],
                                "annotations" : [], "categories" : data["categories"]}
        dataset_seen_unseen = {"info" : data["info"], "images" : [], "licenses" : data["licenses"],
                                "annotations" : [], "categories" : data["categories"]}

        i=1
        for img in data["images"]:
            if img["id"] in imgs:
                dataset_unseen["images"].append(img)
                dataset_seen_unseen["images"].append(img)
            
            if i%10000 == 0:
                print(f"{i} Images processed")

            i+=1

        i=1
        for ann in data["annotations"]:
            if ann["image_id"] in imgs:
                dataset_seen_unseen["annotations"].append(ann)

                if ann["category_id"] in unseen_ids:
                    dataset_unseen["annotations"].append(ann)

            if i%10000 == 0:
                print(f"{i} Annotations processed")

            i+=1

        print(f'Images unseen: {len(dataset_unseen["images"])}')
        print(f'Images seen and unsee: {len(dataset_seen_unseen["images"])}\n')

        print(f'Annotations unseen: {len(dataset_unseen["annotations"])}')
        print(f'Annotations seen and unseen: {len(dataset_seen_unseen["annotations"])}')

        with open(f"/fastdata/vilab14/coco/annotations/instances_val2014_unseen_65_15_new.json", "w") as outfile:
            json.dump(dataset_unseen, outfile, indent=2)
            print("unseen JSON-file saved...")
        with open(f"/fastdata/vilab14/coco/annotations/instances_val2014_seen_unseen_65_15_new.json", "w") as outfile:
            json.dump(dataset_seen_unseen, outfile, indent=2)
            print("seen_unseen JSON-file saved...")

def create_seen_val_set():
    seen_ids = []
    for cls in SEEN_CLASSES:
        seen_ids.append(cls["id"])

    with open(DIR_BASE + "instances_val2014.json") as val_file:
        data = json.load(val_file)

        imgs_with_seen_instances = []
        for ann in data["annotations"]:
            if ann["category_id"] in seen_ids:
                if ann["image_id"] not in imgs_with_seen_instances:
                    imgs_with_seen_instances.append(ann["image_id"])
        print(f"{len(imgs_with_seen_instances)} images with at least one seen instance identified")

        dataset_seen = {"info" : data["info"], "images" : [], "licenses" : data["licenses"],
                                "annotations" : [], "categories" : data["categories"]}

        for img in data["images"]:
            if img["id"] in imgs_with_seen_instances:
                dataset_seen["images"].append(img)

        i=1
        for ann in data["annotations"]:
            if ann["category_id"] in seen_ids:
                assert ann["image_id"] in imgs_with_seen_instances
                dataset_seen["annotations"].append(ann)

            if i%10000 == 0:
                print(f"{i} Annotations processed")

            i+=1

        print(f'Images seen: {len(dataset_seen["images"])}')
        print(f'Annotations seen: {len(dataset_seen["annotations"])}')

        with open(f"/data/vilab16/coco/instances_val2014_seen_65_15_new.json", "w") as outfile:
            json.dump(dataset_seen, outfile, indent=2)
            print("seen JSON-file saved...")


if __name__ == "__main__":
    #create_zsd_and_gzsd_val_sets()
    create_seen_val_set()
