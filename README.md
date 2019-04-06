# Generalized Intersection over Union - PyTorch Faster/Mask R-CNN

Faster/Mask R-CNN with GIoU loss implemented in PyTorch

If you use this work, please consider citing:

```
@article{Rezatofighi_2018_CVPR,
  author    = {Rezatofighi, Hamid and Tsoi, Nathan and Gwak, JunYoung and Sadeghian, Amir and Reid, Ian and Savarese, Silvio},
  title     = {Generalized Intersection over Union},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2019},
}
```

## Modifications in this repository

This repository is a fork of [roytseng-tw/Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch), with an implementation of GIoU and IoU loss while keeping the code as close to the original as possible. It is also possible to train the network with SmoothL1 loss as in the original code. See the options below.

### Losses

The loss can be chosen with the `MODEL.LOSS_TYPE` option in the configuration file. The valid options are currently: `[iou|giou|sl1]`. At this moment, we apply bounding box loss only on final bounding box refinement layer, just as in the paper.

```
MODEL:
  LOSS_TYPE: 'iou'
```

Please take a look at `compute_iou` function of [lib/utils/net.py](lib/utils/net.py) for our GIoU and IoU loss implementation in PyTorch.

### Normalizers

We also implement a normalizer of final bounding box refinement loss. This can be specified with the `MODEL.LOSS_BBOX_WEIGHT` parameter in the configuration file. The default value is `1.0`. We use `MODEL.LOSS_BBOX_WEIGHT` of `10.` for IoU and GIoU experiments.

```
MODEL:
  LOSS_BBOX_WEIGHT: 10.
```

### Network Configurations

We add sample configuration files used for our experiment in `config/baselines`. Our experiments in the paper are based on `e2e_faster_rcnn_R-50-FPN_1x.yaml` and `e2e_mask_rcnn_R-50-FPN_1x.yaml` as following:

```
e2e_faster_rcnn_R-50-FPN_giou_1x.yaml  # Faster R-CNN + GIoU loss
e2e_faster_rcnn_R-50-FPN_iou_1x.yaml   # Faster R-CNN + IoU loss
e2e_mask_rcnn_R-50-FPN_giou_1x.yaml    # Mask R-CNN + GIoU loss
e2e_mask_rcnn_R-50-FPN_iou_1x.yaml     # Mask R-CNN + IoU loss
```

## Train and evaluation commands

For detailed installation instruction and network training options, please take a look at the README file or issue of [roytseng-tw/Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch). Following is a sample command we used for training and testing Faster R-CNN with GIoU.

```
python tools/train_net_step.py --dataset coco2017 --cfg configs/baselines/e2e_faster_rcnn_R-50-FPN_giou_1x.yaml --use_tfboard
python tools/test_net.py --dataset coco2017 --cfg configs/baselines/e2e_faster_rcnn_R-50-FPN_giou_1x.yaml --load_ckpt {full_path_of_the_trained_weight}
```

## Pretrained weights

Here are the trained models using the configurations in this repository.

 - [Faster RCNN + SmoothL1](https://giou.stanford.edu/rcnn_weights/faster_sl1.pth)
 - [Faster RCNN + IoU](https://giou.stanford.edu/rcnn_weights/faster_iou.pth)
 - [Faster RCNN + GIoU](https://giou.stanford.edu/rcnn_weights/faster_giou.pth)
 - [Mask RCNN + SmoothL1](https://giou.stanford.edu/rcnn_weights/mask_sl1.pth)
 - [Mask RCNN + IoU](https://giou.stanford.edu/rcnn_weights/mask_iou.pth)
 - [Mask RCNN + GIoU](https://giou.stanford.edu/rcnn_weights/mask_giou.pth)
