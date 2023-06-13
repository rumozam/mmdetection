import json
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

DIRECTORY_MODIFIED = '/data/vilab16/logs/baseline_modified_coco/yolox_modified_10percent/eval_seen_65'
DIRECTORY = "/data/vilab16/logs/baseline/yolox_s_coco_10percent/eval_seen_65"

def plot__mAP():
    epochs_modified = []
    bbox_mAP_modified = []
    bbox_mAP_50_modified = []
    bbox_mAP_75_modified = []

    prefix = 'eval_epoch_'
    eval_results = dict()
    files = os.listdir(DIRECTORY_MODIFIED)
    for file_path in files:
        if file_path.startswith(prefix):
            print(file_path)
            with open(f'{DIRECTORY_MODIFIED}/{file_path}') as file:
                epoch = file_path[len(prefix):]
                epoch = epoch[:epoch.find('_')]
                epochs_modified.append(int(epoch))
                key = int(epoch)
                value = []

                for line in file:
                    data = json.loads(line)
                    value.append(data['metric']['bbox_mAP'])
                    value.append(data['metric']['bbox_mAP_50'])
                    value.append(data['metric']['bbox_mAP_75'])
                
                eval_results[key] = value

    epochs_modified.sort()
    for epoch in epochs_modified:
        results = eval_results[epoch]
        bbox_mAP_modified.append(results[0])
        bbox_mAP_50_modified.append(results[1])
        bbox_mAP_75_modified.append(results[2])

    epochs = []
    bbox_mAP = []
    bbox_mAP_50 = []
    bbox_mAP_75 = []

    prefix = 'eval_epoch_'
    eval_results = dict()
    files = os.listdir(DIRECTORY)
    for file_path in files:
        if file_path.startswith(prefix):
            print(file_path)
            with open(f'{DIRECTORY}/{file_path}') as file:
                epoch = file_path[len(prefix):]
                epoch = epoch[:epoch.find('_')]
                epochs.append(int(epoch))
                key = int(epoch)
                value = []

                for line in file:
                    data = json.loads(line)
                    value.append(data['metric']['bbox_mAP'])
                    value.append(data['metric']['bbox_mAP_50'])
                    value.append(data['metric']['bbox_mAP_75'])
                
                eval_results[key] = value

    epochs.sort()
    for epoch in epochs:
        results = eval_results[epoch]
        bbox_mAP.append(results[0])
        bbox_mAP_50.append(results[1])
        bbox_mAP_75.append(results[2])

    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.set_ylabel('bbox_mAP_50')
    ax.set_xlabel('epoch')
    line,_ = ax.plot(epochs, bbox_mAP_50, '-', epochs, bbox_mAP_50, 'o', color=mcolors.TABLEAU_COLORS['tab:orange'], label="YOLOX-s small, 10%")
    line_modified,_ = ax.plot(epochs_modified, bbox_mAP_50_modified, '-', epochs_modified, bbox_mAP_50_modified, 'o', color=mcolors.TABLEAU_COLORS['tab:blue'], label="modified")
    ax.legend(handles=[line_modified,line])
    ax.grid()
    ax.set_xlim([0, 305])
    ax.set_ylim([0, 0.35])
    fig.savefig('yolox_s_vs_modified_mAP_50_seen.png')
    print(f'saved yolox_s_vs_modified_mAP_50_seen.png')

if __name__ == "__main__":
    plot__mAP()