import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result

cfg = mmcv.Config.fromfile('local_configs/retinanet_r50_fpn_2x_pretrain_old_resnet50.py')
cfg.model.pretrained = None

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, './work_dirs/retinanet_r50_fpn_2x_pretrain_old_resnet50/epoch_24.pth')

# test a single image
img = mmcv.imread('test.jpg')
result = inference_detector(model, img, cfg)
show_result(img, result)

# test a list of images
imgs = ['1.jpg', '2.jpg','0.jpg', '3.jpg']
for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
    print(i, imgs[i])
    show_result(imgs[i], result)
    # visualize the results in a new window
    # show_result(img, result, model.CLASSES)
    # or save the visualization results to image files
    outname='det_'+imgs[i]
    show_result(img, result, model.CLASSES, out_file=outname)