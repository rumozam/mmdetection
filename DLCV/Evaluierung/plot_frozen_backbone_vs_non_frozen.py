import json
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

DIRECTORY_FROZEN = '/data/vilab16/logs/imagenet/yolox_modified_coco_imagenet_10percent_trained_on_3gpus/eval_unseen/'
DIRECTORY = "/data/vilab16/logs/imagenet/12_02_22/yolox_coco_imagenet_10percent_backbone_not_frozen/"

def plot__mAP():
    epochs_frozen = []
    bbox_mAP_50_frozen = []

    prefix = 'eval_epoch_'
    eval_results = dict()
    files = os.listdir(DIRECTORY_FROZEN)
    for file_path in files:
        if file_path.startswith(prefix):
            print(file_path)
            with open(f'{DIRECTORY_FROZEN}/{file_path}') as file:
                epoch = file_path[len(prefix):]
                epoch = epoch[:epoch.find('_')]
                epochs_frozen.append(int(epoch))
                key = int(epoch)
                value = []

                for line in file:
                    data = json.loads(line)
                    value.append(data['metric']['bbox_mAP_50'])
                
                eval_results[key] = value

    epochs_frozen.sort()
    for epoch in epochs_frozen:
        results = eval_results[epoch]
        bbox_mAP_50_frozen.append(results[0])

    epochs = []
    bbox_mAP_50 = []
    with open(DIRECTORY + "20220212_101300.log.json") as log_file:
        for line in log_file:
            if line.startswith('{"mode": "val"'):
                data = json.loads(line)
                if data['epoch'] <= 185:
                    epochs.append(data['epoch'])
                    bbox_mAP_50.append(data['bbox_mAP_50'])

    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.set_ylabel('bbox_mAP_50')
    ax.set_xlabel('epoch')
    line,_ = ax.plot(epochs, bbox_mAP_50, '-', epochs, bbox_mAP_50, 'o', color=mcolors.TABLEAU_COLORS['tab:orange'], label="non-frozen backbone")
    line_frozen,_ = ax.plot(epochs_frozen, bbox_mAP_50_frozen, '-', epochs_frozen, bbox_mAP_50_frozen, 'o', color=mcolors.TABLEAU_COLORS['tab:blue'], label="frozen backbone")
    ax.legend(handles=[line_frozen,line])
    ax.grid()
    plt.xticks(range(165, 187, 5))
    ax.set_ylim([0, 0.11])
    fig.savefig('frozen_vs_non_frozen_mAP_50_unseen.png')
    print(f'saved frozen_vs_non_frozen_mAP_50_unseen.png')

if __name__ == "__main__":
    plot__mAP()