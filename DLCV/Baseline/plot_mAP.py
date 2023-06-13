from ast import BitXor
import json
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_yolox_s_x_mAP():
    s_epoch = []
    s_bbox_mAP = []
    s_bbox_mAP_50 = []
    s_bbox_mAP_75 = []
    with open("yolox_s_8x8_300e_coco_20211121_095711.log.json") as log_file:
        for line in log_file:
            if line.startswith('{"mode": "val"'):
                data = json.loads(line)

                s_epoch.append(data['epoch'])
                s_bbox_mAP.append(data['bbox_mAP'])
                s_bbox_mAP_50.append(data['bbox_mAP_50'])
                s_bbox_mAP_75.append(data['bbox_mAP_75'])
    assert len(s_epoch) == len(s_bbox_mAP) == len(s_bbox_mAP_50) == len(s_bbox_mAP_75)

    x_epoch = []
    x_bbox_mAP = []
    x_bbox_mAP_50 = []
    x_bbox_mAP_75 = []
    with open("yolox_x_8x8_300e_coco_20211126_140254.log.json") as log_file:
        for line in log_file:
            if line.startswith('{"mode": "val"'):
                data = json.loads(line)

                x_epoch.append(data['epoch'])
                x_bbox_mAP.append(data['bbox_mAP'])
                x_bbox_mAP_50.append(data['bbox_mAP_50'])
                x_bbox_mAP_75.append(data['bbox_mAP_75'])
    assert len(x_epoch) == len(x_bbox_mAP) == len(x_bbox_mAP_50) == len(x_bbox_mAP_75)

    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.set_ylabel('bbox_mAP')
    ax.set_xlabel('epoch')
    ax.set_xlim([0, 310])
    line_s,_ = ax.plot(s_epoch, s_bbox_mAP, '-', s_epoch, s_bbox_mAP, 'o', label="yolox_s", color=mcolors.TABLEAU_COLORS['tab:blue'])
    line_x,_ = ax.plot(x_epoch, x_bbox_mAP, '-', x_epoch, x_bbox_mAP, 'o', label="yolox_x", color=mcolors.TABLEAU_COLORS['tab:red'])
    #ax.legend(handles=[line_x,line_s], bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=1)
    ax.grid()
    plt.tight_layout()
    fig.savefig('yolox_mAP_1.png')

    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.set_ylabel('bbox_mAP_50')
    ax.set_xlabel('epoch')
    ax.set_xlim([0, 310])
    ax.set_ylim([0, 0.71])
    line_s,_ = ax.plot(s_epoch, s_bbox_mAP_50, '-', s_epoch, s_bbox_mAP_50, 'o', label="yolox_s", color=mcolors.TABLEAU_COLORS['tab:blue'])
    line_x,_ = ax.plot(x_epoch, x_bbox_mAP_50, '-', x_epoch, x_bbox_mAP_50, 'o', label="yolox_x", color=mcolors.TABLEAU_COLORS['tab:red'])
    #ax.legend(handles=[line_x,line_s], bbox_to_anchor=(1, 1), loc="upper left", borderaxespad=1)
    ax.grid()
    plt.tight_layout()
    fig.savefig('yolox_mAP_50_1.png')

    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.set_ylabel('bbox_mAP_75')
    ax.set_xlabel('epoch')
    ax.set_xlim([0, 310])
    line_s,_ = ax.plot(s_epoch, s_bbox_mAP_75, '-', s_epoch, s_bbox_mAP_75, 'o', label="yolox_s", color=mcolors.TABLEAU_COLORS['tab:blue'])
    line_x,_ = ax.plot(x_epoch, x_bbox_mAP_75, '-', x_epoch, x_bbox_mAP_75, 'o', label="yolox_x", color=mcolors.TABLEAU_COLORS['tab:red'])
    #ax.legend(handles=[line_x,line_s], bbox_to_anchor=(1, 1), loc="upper left", borderaxespad=1)
    ax.grid()
    plt.tight_layout()
    fig.savefig('yolox_mAP_75_1.png')

if __name__ == "__main__":
    plot_yolox_s_x_mAP()