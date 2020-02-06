import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result
import time



cfg = mmcv.Config.fromfile('../local_configs/retinanet_r50_fpn_2x_pretrain_na_resnet50.py')
cfg.model.pretrained = None

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, '../work_dirs/retinanet_r50_fpn_2x_pretrain_na_resnet50/epoch_24.pth')

"""
# test a single image
# img = mmcv.imread('test.jpg')
# result = inference_detector(model, img, cfg)
# show_result(img, result)

# # test a list of images
# imgs = ['0.jpg', '1.jpg','2.jpg', '3.jpg', '4.jpg','5.jpg', '6.jpg', '7.jpg','8.jpg', '9.jpg', '10.jpg']
# for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
#     print(i, imgs[i])
#     # show_result(imgs[i], result)
#     # visualize the results in a new window
#     # show_result(img, result, model.CLASSES)
#     # or save the visualization results to image files
#     outname='det_'+imgs[i]
#     show_result(imgs[i], result, out_file=outname)
"""


video = mmcv.VideoReader('/home/g1007540910/detection/work_dirs/IMG_0593.mp4')
print(video.__len__())

i=-1
# for frame in video:
#     result = inference_detector(model, frame,cfg)
#     i = i+1
#     print("procesing frame: {}   /   {}".format(i,video.__len__()))
#     show_result(frame, result, out_file="frames/"+str(i).zfill(6)+".jpg")
# mmcv.frames2video('frames', 'result.avi')

st = time.perf_counter()
for frame in video:
    result = inference_detector(model, frame, cfg)
print("Total time: {}".format(time.perf_counter() - st))
