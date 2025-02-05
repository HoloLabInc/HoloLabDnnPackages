# CHANGES

## [1.0.6] - 2025-02-05

- Update Unity Sentis to 2.1.2 from 2.1.1.

## [1.0.5] - 2024-11-15

- Update Unity Sentis to 2.1.1 from 2.1.0.
- Fix GetOutputShapes() to support models with Int, Short, and Byte as data type of input layers.
- Change access modifiers for some variables and functions for more extensibility and flexibility.

## [1.0.4] - 2024-09-13

- Update Unity Sentis to 2.1 from 1.6.
- Remove apply_quantize argument from constructor, and add ApplyQuantize() function as replacement.
- Delete IBackend variable because IBackend API has been deprecated in Unity Sentis 2.0 and later.
- Add SetEditedModel() function for replace with edited model by Functional API.

## [1.0.3] - 2024-08-20

- Update Unity Sentis to 1.6 from 1.5.
- Add support inference split to multiple frames.

## [1.0.2] - 2024-06-12

- Update Unity Sentis to 1.5 from 1.4.
- Update Unity Required Version to 2023.2 (Unity 6) or later.
- Remove Constructor to Load Model from ONNX at Runtime.
- Add Constructor to Load Model from Sentis (*.sentis) at Runtime.
- Add Constructor to Load Model from Stream at Runtime.

## [1.0.1] - 2024-04-10

- Update Unity Sentis to 1.4 from 1.3.

## [1.0.0] - 2024-01-01

### First Release jp.co.hololab.base

- Add support for Unity Package Manager.
- Add bace model for dnn inference.
