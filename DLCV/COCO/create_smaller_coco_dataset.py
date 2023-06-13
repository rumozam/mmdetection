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

PERC = 10

def main():
    dir_base = "data/coco/annotations/"

    # Get array with coco ids of seen classes and create dicts for them
    seen_ids = []

    for cls in SEEN_CLASSES:
        seen_ids.append(cls["id"])

    counts_seen = dict.fromkeys(seen_ids, 0)
    counts_used = dict.fromkeys(seen_ids, 0)

    image_ids = []
    i = 0 
    with open(dir_base + "instances_train2014_seen_65_15.json") as train_file:
        data = json.load(train_file)

        # Create array with image ids
        for img in data["images"]:
            image_ids.append(img["id"])

        # Dict with correspondence between image ids and annotations
        image_classes = dict([(key, []) for key in image_ids])

        for ann in data["annotations"]:
            cat_id = ann["category_id"]
            img_id = ann["image_id"]

            # Count the number of different categories; Necessary to get new dataset with the same distribution
            counts_seen[cat_id] += 1

            image_classes[img_id].append([cat_id, ann["id"]])

            i += 1
            if i%10000 == 0:
                print(i)
        
    # Dict with percentage of class counts
    counts_seen_perc = dict()

    for i in counts_seen.keys():
        counts_seen_perc[i] = int(np.ceil(counts_seen[i]*(PERC/100)))
    
    new_img_ids = []
    new_ann_ids = []

    for key, val in image_classes.items():
        # Dict for class counts in current image
        temp = dict.fromkeys(seen_ids,0)

        for cls, _ in val:
            temp[cls] += 1

        add = True

        # Check if current image would fit in new set or if class count would exceed it
        for i in seen_ids:
            if temp[i] + counts_used[i] > counts_seen_perc[i]:
                add = False

        # If image fits into new set, save image id and annotation ids
        if add:
            for key_temp, val_temp in temp.items():
                counts_used[key_temp] += val_temp

            new_img_ids.append(key)
            for _, ann in val:
                new_ann_ids.append(ann)


    # Dict for the new data set
    new_dataset = {"images" : [], "annotations" : [], "categories" : data["categories"]}

    # Add images to new data set
    for img in data["images"]:
        if img["id"] in new_img_ids:
            new_dataset["images"].append(img)

    print("Images saved...")
    print(f"{len(new_img_ids)} Images ({float(len(new_img_ids))/len(data['images'])})%\n")

    # Add annotations to new data set
    for ann in data["annotations"]:
        if ann["id"] in new_ann_ids:
            new_dataset["annotations"].append(ann)

    print("Annotations saved...")
    print(f"{len(new_ann_ids)} Annotations ({float(len(new_ann_ids))/len(data['annotations'])})%\n")

    # Save new data set in json file
    with open(f"/fastdata/vilab14/coco/annotations/instances_train2014_seen_65_15_percent_{PERC}.json", "w") as outfile:
        json.dump(new_dataset, outfile)
        print("JSON-file saved...")

if __name__ == "__main__":
    main()