import json
import numpy as np

PERC = 20

def main():
    with open("../../data/imagenet_annotations/imagenet2012_annotations_seen_65_15.json") as train_file:
        data = json.load(train_file)

        # Create array with image ids
        image_ids_dict = dict()
        for img in data["images"]:
            image_ids_dict[img["id"]] = img

        # Create array with category ids
        category_ids = []
        for cat in data["categories"]:
            category_ids.append(cat["id"])

        counts_seen = dict.fromkeys(category_ids, 0)
        counts_used = dict.fromkeys(category_ids, 0)

        # Dict with correspondence between image ids and annotations
        image_ids = list(image_ids_dict.keys())
        image_classes = dict([(key, []) for key in image_ids])

        ann_ids_dict = dict()
        for ann in data["annotations"]:
            cat_id = ann["category_id"]
            img_id = ann["image_id"]
            ann_ids_dict[ann["id"]] = ann

            # Count the number of different categories; Necessary to get new dataset with the same distribution
            counts_seen[cat_id] += 1
            image_classes[img_id].append([cat_id, ann["id"]])

    print("Retrieved images, categories and annotations")
        
    # Dict with percentage of class counts
    counts_seen_perc = dict()
    for i in counts_seen.keys():
        counts_seen_perc[i] = int(np.ceil(counts_seen[i]*(PERC/100)))
    
    new_img_ids = []
    new_ann_ids = []

    for image_id, val in image_classes.items():
        if len(val) == 0:
            # image belongs to unseen category
            continue
        
        # there should be only 1 category and 1 annotation per image
        assert len(val) == 1
        [cat_id, ann_id] = val[0]
        add = True

        # Check if current image would fit in new set or if class count would exceed it
        if counts_used[cat_id] + 1 > counts_seen_perc[cat_id]:
                add = False

        # If image fits into new set, save image id and annotation ids
        if add:
            counts_used[cat_id] += 1
            new_img_ids.append(image_id)
            new_ann_ids.append(ann_id)

    assert len(new_img_ids) == len(new_ann_ids)

    # Dict for the new data set
    new_dataset = {"images" : [], "annotations" : [], "categories" : data["categories"]}

    # Add images to new data set
    for img_id in new_img_ids:
        new_img = image_ids_dict[img_id]
        new_dataset["images"].append(new_img)

    print("Images saved...")
    print(f"{len(new_img_ids)} Images ({float(len(new_img_ids))/len(data['images'])})%\n")

    # Add annotations to new data set
    for ann_id in new_ann_ids:
        new_ann = ann_ids_dict[ann_id]
        new_dataset["annotations"].append(new_ann)

    print("Annotations saved...")
    print(f"{len(new_ann_ids)} Annotations ({float(len(new_ann_ids))/len(data['annotations'])})%\n")

    # Save new data set in json file
    with open(f"../../data/imagenet_annotations/imagenet2012_annotations_seen_65_15_percent_{PERC}.json", "w") as outfile:
        json.dump(new_dataset, outfile)
        print("JSON-file saved...")

if __name__ == "__main__":
    main()