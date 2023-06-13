import json
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

DIRECTORY = '/data/vilab16/logs/baseline/yolox_s_coco_10percent/eval_seen'
DIRECTORY_MODIFIED = '/data/vilab16/logs/baseline_modified_coco/yolox_modified_10percent/eval_seen_smaller_obj_thr'

def plot__recall():
    epochs_modified = []
    bbox_recall_modified = []

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
                    value.append(data['metric']['bbox_AR@100'])
                
                eval_results[key] = value

    epochs_modified.sort()
    for epoch in epochs_modified:
        results = eval_results[epoch]
        bbox_recall_modified.append(results[0])

    epochs = []
    bbox_recall = []

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
                    value.append(data['metric']['bbox_AR@100'])
                
                eval_results[key] = value

    epochs.sort()
    for epoch in epochs:
        results = eval_results[epoch]
        bbox_recall.append(results[0])

    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.set_ylabel('AR@100 (IoU 0.5)')
    ax.set_xlabel('epoch')
    ax.set_ylim([0, 0.6])
    line_modified,_ = ax.plot(epochs_modified, bbox_recall_modified, 'b-', epochs_modified, bbox_recall_modified, 'o', color=mcolors.TABLEAU_COLORS['tab:blue'], label="modified")
    line,_ = ax.plot(epochs, bbox_recall, 'b-', epochs, bbox_recall, 'o', color=mcolors.TABLEAU_COLORS['tab:orange'], label="YOLOX-s small, 10%")
    ax.legend(handles=[line_modified,line])
    ax.grid()

    fig.savefig('recall_yolox_s_small_10p_vs_modified.png')
    print(f'saved recall_yolox_s_small_10p_vs_modified.png')

if __name__ == "__main__":
    plot__recall()