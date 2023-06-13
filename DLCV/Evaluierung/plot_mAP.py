import json
import os
import matplotlib.pyplot as plt

DIRECTORY = '/data/vilab16/logs/yolox_modified_coco_baseline/eval_seen'

def plot__mAP():
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
    ax.set_ylabel('bbox_mAP')
    ax.set_xlabel('epoch')
    ax.plot(epochs, bbox_mAP, 'b-', epochs, bbox_mAP, 'o')
    ax.grid()
    fig.savefig(f'{DIRECTORY}/yolox_s_mAP.png')
    print(f'saved {DIRECTORY}/yolox_s_mAP.png')

    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.set_ylabel('bbox_mAP_50')
    ax.set_xlabel('epoch')
    ax.plot(epochs, bbox_mAP_50, 'b-', epochs, bbox_mAP_50, 'o')
    ax.grid()
    fig.savefig(f'{DIRECTORY}/yolox_s_mAP_50.png')
    print(f'saved {DIRECTORY}/yolox_s_mAP_50.png')

    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.set_ylabel('bbox_mAP_75')
    ax.set_xlabel('epoch')
    ax.plot(epochs, bbox_mAP_75, 'b-', epochs, bbox_mAP_75, 'o')
    ax.grid()
    fig.savefig(f'{DIRECTORY}/yolox_s_mAP_75.png')
    print(f'saved {DIRECTORY}/yolox_s_mAP_75.png')

if __name__ == "__main__":
    plot__mAP()