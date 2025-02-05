# CHANGES

## [1.0.6] - 2025-02-05

- Update Unity Sentis to 2.1.2 from 2.1.1.

## [1.0.5] - 2024-11-15

- Update Unity Sentis to 2.1.1 from 2.1.0.

## [1.0.4] - 2024-09-13

- Update Unity Sentis to 2.1 from 1.6.
- Move some internal post processing to Functional API from IBackend API.
- Remove apply_quantize argument from constructor. Instead of that, please use ApplyQuantize() function inherited from BaseModel class.

## [1.0.3] - 2024-08-20

- Update Unity Sentis to 1.6 from 1.5.

## [1.0.2] - 2024-06-12

- Update Unity Sentis to 1.5 from 1.4.
- Update Unity Required Version to 2023.2 (Unity 6) or later.
- Remove Constructor to Load Model from ONNX at Runtime.
- Add Constructor to Load Model from Sentis (*.sentis) at Runtime.
- Add Constructor to Load Model from Stream at Runtime.

## [1.0.1] - 2024-04-10

- Update Unity Sentis to 1.4 from 1.3.

## [1.0.0] - 2024-01-01

### First Release jp.co.hololab.dnn.depthestimation

- Add support for Unity Package Manager.
- Add depth estimation using MiDaS.
- Add sample scene for depth estimation.
