import torch
import clip
import pickle


device = "cuda" if torch.cuda.is_available() else "cpu"

# model, preprocess = clip.load("ViT-B/32", device=device)
# embedding_dim = 512
# pkl_file = "../../mmdet/label_embeddings_imagenet_tuned_vitb32.pkl"
# txt_file = 'imagenet2012_tuned_labels_text.txt'

# model, preprocess = clip.load("ViT-B/32", device=device)
# embedding_dim = 512
# pkl_file = "../../mmdet/label_embeddings_imagenet_untuned_vitb32.pkl"
# txt_file = '../../data/imagenet_annotations/imagenet2012_labels_text.txt'

model, preprocess = clip.load("ViT-L/14@336px", device=device)
embedding_dim = 768
pkl_file = "../../mmdet/label_embeddings_imagenet_tuned_vitl14_336px.pkl"
txt_file = 'imagenet2012_tuned_labels_text.txt'

# model, preprocess = clip.load("ViT-L/14@336px", device=device)
# embedding_dim = 768
# pkl_file = "../../mmdet/label_embeddings_imagenet_untuned_vitl14_336px.pkl"
# txt_file = '../../data/imagenet_annotations/imagenet2012_labels_text.txt'

categories = []
N_cat = 1000

with open(txt_file,'r') as f:
    name = f.readline()[:-1]

    while name != "":
        categories.append(name)
        name = f.readline()[:-1]

assert(len(categories)==N_cat)
d = torch.zeros((embedding_dim, N_cat))
a = True
for i in range(N_cat):
    category = categories[i]


    prompts = []    

    prompts.append(clip.tokenize("itap of a " + category + ".").to(device))
    prompts.append(clip.tokenize("a bad photo of the " + category + ".").to(device))
    prompts.append(clip.tokenize("a origami " + category + ".").to(device))
    prompts.append(clip.tokenize("a photo of the large " + category + ".").to(device))
    prompts.append(clip.tokenize("a " + category + " in a video game.").to(device))
    prompts.append(clip.tokenize("art of the " + category + ".").to(device))
    prompts.append(clip.tokenize("a photo of the small " + category + ".").to(device))

    if a:
        print(len(category))
        print("itap of a " + category + ".")
        print("a origami " + category + ".")
        print("a photo of the large " + category + ".")
        print("a " + category + " in a video game.")
        print("art of the " + category + ".")
        print("a bad photo of the " + category + ".")
        print("a photo of the small " + category + ".")

    with torch.no_grad():
        text_features = torch.zeros((1, embedding_dim)).to(device)
        for prompt in prompts:
            text_features += model.encode_text(prompt)
        text_features = text_features / (7) # take mean over all prompts
        text_features = text_features / torch.linalg.norm(text_features) # cosine similarity requires normed vectors
        d[:, i] = text_features
        if i%10==0:
            print(f"Class {i} embedded")

    a=False


with open(pkl_file, 'wb') as handle:
    pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

