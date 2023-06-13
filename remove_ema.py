import torch


checkpnt = "../../logs/yolox_modified/epoch_300.pth"

# -> see .dev_scripts/gather_models.py process_checkpoint()
def remove_ema_from_state_dict(checkpnt):
    modified_checkpoint_path = checkpnt[:-4] + '_modified.pth'
    checkpoint = torch.load(checkpnt, map_location='cpu')
    for key in list(checkpoint['state_dict']):
            if key.startswith('ema_'):
                checkpoint['state_dict'].pop(key)
    torch.save(checkpoint, modified_checkpoint_path, _use_new_zipfile_serialization=False)
    print("saved checkpoint without ema to " + modified_checkpoint_path)

if __name__ == '__main__':
    remove_ema_from_state_dict(checkpnt)