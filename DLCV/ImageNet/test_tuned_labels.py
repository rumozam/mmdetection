import clip
import torch
import os
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
imagenet_labels = "imagenet2012_tuned_labels_text.txt"

def get_tuned_labels():
    tuned_labels = []
    untuned = []
    with open("imagenet2012_tuned_labels_text.txt",'r') as labels:
        name = labels.readline()[:-1]

        while name != "":
            if "(" in name:
                tuned_labels.append(name)

                end = name.find("(") -1
                untuned.append(name[:end])

            name = labels.readline()[:-1]
    return tuned_labels, untuned

def get_dir_for_labels(tuned, untuned):
    directories = []
    with open("imagenet2012_id_dir_name.txt",'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            entries = line.split()
            dir = entries[1]
            label = entries[2]

            if len(entries) > 3:
                for i in range(3, len(entries)):
                    label += " " + entries[i]

            if label in untuned:
                directories.append(dir)
            elif label in tuned:
                directories.append(dir)

    assert len(directories) == len(tuned)
    return directories

def get_better_label(label1, label2, dir):
    text_label1 = clip.tokenize(label1).to(device)
    text_label2 = clip.tokenize(label2).to(device)
    text_features_1 = model.encode_text(text_label1)
    text_features_2 = model.encode_text(text_label2)

    # cosine similarity requires normed vectors
    text_features_1 = text_features_1 / torch.linalg.norm(text_features_1)
    text_features_2 = text_features_2 / torch.linalg.norm(text_features_2)
    cosine_sim_1 = 0
    cosine_sim_2 = 0

    dir_full_path = "/fastdata/ILSVRC2012/train/" + dir
    directory = os.fsencode(dir_full_path)
    num_images = 0

    # calculate cosine similarity between image embedding and 
    # text embedding for every image in the directory
    with torch.no_grad():
        for img in os.listdir(directory):
            image_path = dir_full_path + "/" + os.fsdecode(img)
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            image_features = model.encode_image(image)

            image_features = image_features / torch.linalg.norm(image_features)
            cosine_sim_1 += text_features_1 @ image_features.reshape(-1,1)
            cosine_sim_2 += text_features_2 @ image_features.reshape(-1,1)
            num_images += 1

        mean_cosine_sim_1 = cosine_sim_1 / num_images
        mean_cosine_sim_2 = cosine_sim_2 / num_images

        if mean_cosine_sim_1 > mean_cosine_sim_2:
            return label1, mean_cosine_sim_1 - mean_cosine_sim_2
        else:
            return label2, mean_cosine_sim_2 - mean_cosine_sim_1


if __name__ == "__main__":
    tuned_labels, untuned = get_tuned_labels()
    directories = get_dir_for_labels(tuned_labels, untuned)
    better_labels = []

    for i in range(len(tuned_labels)):
        label, diff = get_better_label(tuned_labels[i], untuned[i], directories[i])
        line = label + " %.3f" % diff.item()
        better_labels.append(line)
        print(line)

    with open("better_label_out_of_tuned_and_untuned.txt",'w') as f:
        f.writelines(s + '\n' for s in better_labels)