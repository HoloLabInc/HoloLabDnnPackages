# HoloLab DNN Packages

## About

The HoloLab DNN Packages providing image recognition library using deep learning running on the Unity.  
This packages is implemented based on Unity Sentis, and can be deployed to all platforms supported by Unity.  

## Packages

The HoloLab DNN Packages consists of some packages.  
You can use this library by combination of one base package and some task packages.  
Please see README of each package about how to use package.  

### Base Package

* [jp.co.hololab.dnn.base](packages/jp.co.hololab.dnn.base/Documentation/README.md)  
  This package is providing base class for dnn inference.  
  This package is used as internal dependency for other packages.  
  You can create inference classes for your models using this base class too.  

### Task Packages

* [jp.co.hololab.dnn.classification](packages/jp.co.hololab.dnn.classification/Documentation/README.md)  
  This package is providing classification class using general classification models.  
  This package contains simple inference sample app to learn how to use classification class.  
  ![classification](images/classification.png)  

* [jp.co.hololab.dnn.objectdetection](packages/jp.co.hololab.dnn.objectdetection/Documentation/README.md)  
  This package is providing object detection class using YOLOX model, YOLOv9 MIT model, and DETR based models.  
  This package contains simple inference sample app to learn how to use object detection class.  
  ![objectdetection](images/objectdetection.png)  

* [jp.co.hololab.dnn.depthestimation](packages/jp.co.hololab.dnn.depthestimation/Documentation/README.md)  
  This package is providing depth estimation class using general relative depth estimation models.  
  This package contains simple inference sample app to learn how to use depth estimation class.  
  ![depthestimation](images/depthestimation.png)  

* [jp.co.hololab.dnn.segmentation](packages/jp.co.hololab.dnn.segmentation/Documentation/README.md)  
  This package is providing segmentation class using general segmentation models.  
  This package contains simple inference sample app to learn how to use segmentation class.  
  ![segmentation](images/segmentation.png)  

## Environment

These packages work on Unity 2023.2 (Unity 6) or later.  

## License

Copyright &copy; 2024 [HoloLab Inc.](https://hololab.co.jp/)  
All packages are distributed under the MIT license for now.  
Please see the license for each package.  
