import nltk
from nltk.corpus import wordnet as wn

def get_parents(synset):
    return synset.hypernyms()

def main():
    labels = []
    new_labels = []
    with open('/fastdata/vires01/imagenet_annotations/imagenet2012_labels_text.txt','r') as labels_txt:
        name = labels_txt.readline()[:-1]
    
        while name != "":
            if " " in name:
                name = name.replace(" ", "_")
            labels.append(name)
            name = labels_txt.readline()[:-1]

    for label in labels:
        label_synsets = wn.synsets(label)
        if len(label_synsets) == 0:
            label = label

        if len(label_synsets) > 0:
            grand_parents = []
            p = ""
            g = ""
            for synset in label_synsets:
                parents = get_parents(synset)
                for parent in parents:
                    p += " " + parent._name
                    grand_parents += get_parents(parent)
            label += " (" + p + ")"

            for gp in grand_parents:
                g += " " + gp._name
            label += " (" + g + ")"
        
        new_labels.append(label + "\n")
    
    assert len(new_labels) == 1000
    with open('imagenet2012_labels_text_with_hypernyms.txt', 'w') as file:
        file.writelines(new_labels)

if __name__ == "__main__":
    main()
