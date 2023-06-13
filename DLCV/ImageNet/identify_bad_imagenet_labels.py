import clip
import torch
import os
from PIL import Image
import numpy as np
import pickle as pkl

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
imagenet_labels = '/fastdata/vires01/imagenet_annotations/imagenet2012_labels_text.txt'
imagenet_text_embeddings = '../../mmdet/label_embeddings_imagenet_untuned_vitb32.pkl'

def get_labels():
    labels = []
    with open(imagenet_labels,'r') as f:
        name = f.readline()[:-1]

        while name != "":
            labels.append(name)
            name = f.readline()[:-1]
    assert len(labels) == 1000
    return labels

def get_text_embeddings():
    with open(imagenet_text_embeddings,'rb') as handle:
        label_embeddings_imagenet = pkl.load(handle).to(device)
    assert label_embeddings_imagenet.shape[1] == 1000
    return label_embeddings_imagenet

def get_directories():
    directories = []
    with open("imagenet2012_id_dir_name.txt",'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            entries = line.split()
            dir = entries[1]
            directories.append(dir)

    assert len(directories) == 1000
    return directories


if __name__ == "__main__":
    labels = get_labels()
    text_embeddings = get_text_embeddings()
    directories = get_directories()

    mean_cos_sim = []
    for i in range(1000):
        print(i)
        text_features = text_embeddings[:,i].reshape(1,-1)

        dir_full_path = "/fastdata/ILSVRC2012/train/" + directories[i]
        directory = os.fsencode(dir_full_path)
        
        num_images = 0
        cosine_sim = 0

        # calculate cosine similarity between image embedding and 
        # text embedding for every image in the directory
        with torch.no_grad():
            for img in os.listdir(directory):
                image_path = dir_full_path + "/" + os.fsdecode(img)
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                image_features = model.encode_image(image)
                image_features = image_features / torch.linalg.norm(image_features)
                cosine_sim += text_features @ image_features.float().reshape(-1,1)
                num_images += 1

            mean_cos_sim.append(cosine_sim.item() / num_images)
            #print(label + " " + str(mean_cosine_sim.item()))

    ind = np.argsort(mean_cos_sim)
    sorted_labels = [labels[i] for i in ind]
    sorted_cos_sim = [mean_cos_sim[i] for i in ind]

    lines = []
    for i in range(1000):
        lines.append(sorted_labels[i] + " " + str(sorted_cos_sim[i]))
    

    with open("imagenet_labels_cosine_sim_with_images.txt",'w') as f:
        f.writelines(l + '\n' for l in lines)
