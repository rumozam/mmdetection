import json

all_classes_path = "../data/imagenet_annotations/imagenet2012_labels_text.txt"

unseen_classes_path = "./ImageNet/imagenet_48_17_unseen_classes.json"

start_lines_at = 0

with open(all_classes_path) as file:
    lines = file.readlines()

with open(unseen_classes_path) as file:
    unseen_classes = json.load(file)

unknown_indices = []

for number, line in enumerate(lines, start_lines_at):
    for name in unseen_classes:
        if name == line[:-1]:
            unknown_indices.append(number)
            unseen_classes.remove(name)
            print('Class ' + name + ' is in line ' + str(number))

print(unknown_indices)

if len(unseen_classes) > 0:
    print("The following classes were not found. Look for an error:")
    [print(name) for name in unseen_classes]