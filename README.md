# ASCNet
ASCNet:3D object detection from point cloud  based on adaptive spatial context features

![](https://github.com/penghao1990/ASCNet/blob/main/doc/ASCNet.png)

## Results on KITTI test set

[Submission link](http://www.cvlibs.net/datasets/kitti/eval_object_detail.php?&result=2ebb102bbe17b89131b8997bfc5e259910428e70)
![](https://github.com/penghao1990/ASCNet/blob/main/doc/KITTI_test.png {height="50%" width="50%"})

## Results on KITTI val split
```
Car AP@0.70, 0.70, 0.70:
bbox AP:90.8081, 89.8742, 89.3808
bev  AP:90.3067, 88.1557, 87.3697
3d   AP:89.1179, 79.2503, 78.5827
aos  AP:90.75, 89.70, 89.12
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:96.6679, 95.2487, 92.9368
bev  AP:93.5881, 89.5308, 89.1006
3d   AP:90.1054, 83.3291, 80.8298
aos  AP:96.59, 95.03, 92.63

Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:61.4309, 58.6835, 54.1988
bev  AP:57.5988, 51.0879, 49.2069
3d   AP:54.9196, 48.3906, 44.0727
aos  AP:41.54, 40.56, 37.60
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:61.3321, 56.9168, 53.6227
bev  AP:56.4553, 49.8571, 46.3196
3d   AP:52.7456, 45.6248, 42.0676
aos  AP:38.51, 36.41, 34.11

Cyclist AP@0.50, 0.50, 0.50:
bbox AP:88.2491, 75.7573, 73.3316
bev  AP:86.1377, 70.2592, 64.6622
3d   AP:84.3397, 65.5741, 63.3771
aos  AP:87.42, 73.21, 70.69
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:92.4393, 78.6260, 74.1377
bev  AP:89.8029, 69.7474, 65.2652
3d   AP:87.6766, 67.2689, 62.7271
aos  AP:91.54, 75.75, 71.25

```

# Acknowledgement
Our code is based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). Thanks OpenMMLab Development Team for their awesome codebases.
