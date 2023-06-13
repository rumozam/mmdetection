import json

all_classes_path = "../../data/coco/annotations/instances_val2014.json"

# unseen_classes_path = "coco_65_15_unseen_classes.json"
# unseen_classes_path = "coco_48_17_seen_classes.json"
unseen_classes_path = "coco_48_17_unseen_classes.json"


all_classes = []
with open(all_classes_path) as file:
    data = json.load(file)
    categories = data["categories"]
    for cat in categories:
        all_classes.append(cat["name"])

with open(unseen_classes_path) as file:
    unseen_classes = json.load(file)

indices_unseen = []
for cat in unseen_classes:
    idx = all_classes.index(cat)
    indices_unseen.append(idx)

indices_unseen.sort()
print(indices_unseen)