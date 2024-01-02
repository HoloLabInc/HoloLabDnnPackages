# README

## About

This package is provides base model for dnn inference.  
This base model is implemented based on Sentis.  

## Environment

These packages works on Unity 2021.3 LTS or later.  

## License

Copyright &copy; 2024 [HoloLab Inc.](https://hololab.co.jp/)  
Distributed under the [MIT License](LICENSE).  

## How To Add Package

This package is used as internal dependency for other packages.  
Please add this package with any packages that depend on this package.  

## How To Use



## Note

This package depend on BaseModel/PreProcess shader of jp.co.hololab.dnn.base.  
Please add BaseModel/PreProcess shader to Always Included Shaders before building your application.  
This setting is automatically by editor extension when importing this package.  

1. [Edit]>[Project Settings]
2. [Graphics]>[Always Included Shaders]
3. increment size and set BaseModel/PreProcess shader to new element

![Always Included Shaders](image.png)
