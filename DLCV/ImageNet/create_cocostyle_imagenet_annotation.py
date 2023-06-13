import json
import os
from PIL import Image


def get_coco_oversample_ratio():
    with open("../../data/imagenet_annotations/imagenet2012_annotations_seen_65_15.json","r") as imnet:
        imagenet = json.load(imnet)
        imagenet_images = len(imagenet["images"])

    with open("../../data/coco/annotations/instances_train2014_seen_65_15.json", "r") as coco:
        coco = json.load(coco)
        coco_images = len(coco["images"])

    print(f"Images Imagenet: {imagenet_images}")
    print(f"Images COCO: {coco_images}")

    print(f"oversample coco for 1:4 ratio: {imagenet_images/(4*coco_images)}")

def create_label_classname_directory_correspondance():
    with open("../../data/imagenet_annotations/imagenet2012_labels_text.txt", "r") as classnamefile:
        with open("imagenet2012_labels.txt", "r") as classdirectoryfile:
            out = open("imagenet2012_id_dir_name.txt", "a")
            name = classnamefile.readline()[:-1]
            dir = classdirectoryfile.readline()[:-1]

            i=0
            while name != "" and dir != "":
                out.write(str(i) + " " + dir + " " + name + "\n")

                name = classnamefile.readline()[:-1]
                dir = classdirectoryfile.readline()[:-1]
                i+=1
            out.close()



def main():
    #unseen_classes_path = "imagenet_65_15_unseen_classes.json"
    #annotations_path = "../../data/imagenet_annotations/imagenet2012_annotations_seen_65_15.json"

    unseen_classes_path = "imagenet_48_17_unseen_classes.json"
    annotations_path = "../../data/imagenet_annotations/imagenet2012_annotations_seen_48_17.json"

    imagenet_dataset = {"images" : [], "annotations" : [], "categories" : []}

    unseen_ids = []
    with open(unseen_classes_path,"r") as f:
        unseen_classes = json.load(f)

    # Add classnames to dataset
    with open("../../data/imagenet_annotations/imagenet2012_labels_text.txt", "r") as classnamefile:
        line = classnamefile.readline()[:-1]

        i=0
        while line != "":
            imagenet_dataset["categories"].append({"id" : i, "name" : line})
            if line in unseen_classes:
                print(f"Class {i} : {line} is unseen")
                unseen_ids.append(i)
            
            line = classnamefile.readline()[:-1]
            i+=1
        print(f'{len(unseen_ids)} unseen classes')
        assert len(unseen_classes) == len(unseen_ids)
    
    # Add images and annotations to dataset
    with open("imagenet2012_labels.txt", "r") as classdirectoryfile:
        dataroot = "../../data/imagenet/train"
        line = classdirectoryfile.readline()[:-1]

        class_id = 0
        image_id = 0
        while line != "":
            # Dont add images from unseen classes
            if class_id in unseen_ids:
                print(f"Don't add class {class_id}")
                line = classdirectoryfile.readline()[:-1]
                class_id+=1
                continue

            image_path = os.walk(dataroot+"/"+line)

            for _, _, files in image_path:
                for filename in files:
                    img = Image.open(dataroot+"/"+line+"/"+filename)
                    w,h = img.size
                    imagenet_dataset["images"].append({"file_name" : line+"/"+filename, "height" : h, "width" : w, "id" : image_id})
                    # Add None as bbox; Later in training, we can use this to decide which losses we backpropagate
                    # Also, we can use image_id for annotation id, since we only have one annotation per image
                    imagenet_dataset["annotations"].append({"segmentation" : None, "area" : None, "image_id" : image_id, "bbox" : None, "category_id" : class_id, "id" : image_id})
                    image_id += 1

            if class_id % 10 == 0:
                print(f"Class {class_id} added")

            line = classdirectoryfile.readline()[:-1]
            class_id += 1
    
    if unseen_classes != []:
        # Save new data set in json file
        print("Saving only seen classes")
        
        with open(annotations_path, "w") as outfile:
            json.dump(imagenet_dataset, outfile,indent=2)
            print("JSON-file for seen classes saved...")
        
    else:
        print("Saving all classes")
        
        # Save new data set in json file
        with open("../../data/imagenet_annotations/imagenet2012_annotations.json", "w") as outfile:
            json.dump(imagenet_dataset, outfile,indent=2)
            print("JSON-file for all classes saved...")
        

if __name__ == "__main__":
    #get_coco_oversample_ratio()
    #create_label_classname_directory_correspondance()
    main()