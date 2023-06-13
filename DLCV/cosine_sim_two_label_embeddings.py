import torch
import clip

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

text_1 = clip.tokenize("fly").to(device)
text_2 = clip.tokenize("airplane").to(device)

with torch.no_grad():
    text_features_1 = model.encode_text(text_1)
    text_features_2 = model.encode_text(text_2)

    text_features_1 = text_features_1 / torch.linalg.norm(text_features_1)
    text_features_2 = text_features_2 / torch.linalg.norm(text_features_2)

    cosine_sim = text_features_1 @ text_features_2.T
    print(cosine_sim)
