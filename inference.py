from mmdet.apis import init_detector, inference_detector
import torch

model = 'yolox_tiny_8x8_300e_coco.py'

config_file = './configs/yolox/' + model
checkpoint = './work_dirs/' + model[:-3] + '/latest.pth'

img = 'test_image.jpg'
result_file = './work_dirs/' + model[:-3] + '/inference_result.jpg'

# remove ema_ from state dict
# -> see .dev_scripts/gather_models.py process_checkpoint()
modified_checkpoint = checkpoint[:-4] + '_modified.pth'
checkpoint = torch.load(checkpoint, map_location='cpu')
for key in list(checkpoint['state_dict']):
        if key.startswith('ema_'):
            checkpoint['state_dict'].pop(key)
torch.save(checkpoint, modified_checkpoint, _use_new_zipfile_serialization=False)

# for debugging only
# print(checkpoint['meta']['CLASSES'])

# build the model from a config file and a checkpoint file
model = init_detector(config_file, modified_checkpoint)

# test a single image
result = inference_detector(model, img)

# save the results to out_file
model.show_result(img, result, out_file=result_file)