import os
import json
import torch
#import clip
import tensorflow as tf
from tensorboard.plugins import projector

# Set up a logs directory, so Tensorboard knows where to look for files.
log_dir='./logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Save Labels separately on a line-by-line manner.
unseen = ["airplane", "train", "parking meter", "cat", "bear", "suitcase", "frisbee", "snowboard", "fork", "sandwich", "hot dog", "toilet", "mouse", "toaster", "hair drier"]
assert len(unseen) == 15
with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
    for word in unseen:
        f.write("{}\n".format(word))

# create embedding matrix only for unseen
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

d = torch.zeros((15,512))
for i in range(15):
    category = unseen[i]
    print(category)

    prompts = []
    prompts.append(clip.tokenize("itap of a " + category + ".").to(device))
    prompts.append(clip.tokenize("a bad photo of the " + category + ".").to(device))
    prompts.append(clip.tokenize("a origami " + category + ".").to(device))
    prompts.append(clip.tokenize("a photo of the large " + category + ".").to(device))
    prompts.append(clip.tokenize("a " + category + " in a video game.").to(device))
    prompts.append(clip.tokenize("art of the " + category + ".").to(device))
    prompts.append(clip.tokenize("a photo of the small " + category + ".").to(device))

    with torch.no_grad():
        text_features = torch.zeros((1, 512))
        for prompt in prompts:
            text_features += model.encode_text(prompt)
    text_features = text_features / 7 # take mean over all prompts
    text_features = text_features / torch.linalg.norm(text_features) # cosine similarity requires normed vectors
    d[i,:] = text_features


embedding_var = tf.Variable(d)
checkpoint = tf.train.Checkpoint(embedding=embedding_var)
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = './metadata.tsv'
projector.visualize_embeddings(log_dir, config)