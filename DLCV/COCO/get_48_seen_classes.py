import json

classes_path = "../../data/coco/annotations/instances_train2014_seen_48_17.json"

classes = []
with open(classes_path) as file:
    data = json.load(file)
    categories = data["categories"]
    for cat in categories:
        classes.append(cat["name"])

with open(f"coco_48_17_seen_classes.json", "w") as outfile:
    json.dump(classes, outfile, indent=2)