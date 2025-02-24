# CHANGES

## [1.0.6] - 2025-02-05

- Update Unity Sentis to 2.1.2 from 2.1.1.

## [1.0.5] - 2024-11-15

- Update Unity Sentis to 2.1.1 from 2.1.0.
- Add object detection using DETR based models with "images"(1x3xHxW) , "orig_target_sizes"(1x2) at input layers, and "labels"(1xNUM_DETECT), "boxes"(1xNUM_DETECTx4), "scores"(1xNUM_DETECT) at output layers. (e.g. RT-DETR, D-FINE)
- Rename structure for detection results to BoundingBox from Object to avoid conflicts with System.Object and Unity.Object. This breaking changes will be little impact on user code, because generally used type inference.

## [1.0.4] - 2024-09-13

- Update Unity Sentis to 2.1 from 1.6.
- Move some internal post processing to Functional API from IBackend API.
- Remove apply_quantize argument from constructor. Instead of that, please use ApplyQuantize() function inherited from BaseModel class.
- Add DrawLabel() function for TextMeshPro.

## [1.0.3] - 2024-08-20

- Update Unity Sentis to 1.6 from 1.5.
- Add object detection using YOLOv9 MIT.
- Fix bug about continuous processing for multiple frames.
- Fix bug about draw object when set stretch to RectTracnsform of graphic object.

## [1.0.2] - 2024-06-12

- Update Unity Sentis to 1.5 from 1.4.
- Update Unity Required Version to 2023.2 (Unity 6) or later.
- Remove Constructor to Load Model from ONNX at Runtime.
- Add Constructor to Load Model from Sentis (*.sentis) at Runtime.
- Add Constructor to Load Model from Stream at Runtime.

## [1.0.1] - 2024-04-10

- Update Unity Sentis to 1.4 from 1.3.

## [1.0.0] - 2024-01-01

### First Release jp.co.hololab.dnn.objectdetection

- Add support for Unity Package Manager.
- Add object detection using YOLOX.
- Add sample scene for object detection.
