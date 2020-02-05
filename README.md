# UI2CODE - A Computer Vision Based Reverse Engineering of Graphical User Interface

## What is it?

UI2CODE is a system converting the GUI image into cooresponding front-end code that achieves the same visual effect and expected functionality of the input GUI.

It comprises two major parts: 
* UI components detection: localize and classify all UI elements on the given image
  * Graphical components detection 
  * Text recognition through CTPN 
* Code generation
  * DOM tree construction
  * HTML + CSS generation
   
## File structure
*config/*
* Set path of the CNN training result (UI components classification) and CTPN training result (text recognition) 
* Set parameter for graphical components detection 

*ctpn/*
* CTPN implementation

*data/*
* Input image and outputs

*uied/*
* Graphical UI elemnts localization
* Graphical UI elemnts classification by CNN

*main.py*
* Process a batch of images continuously 

*main_single.py*
* Process a signle image

*merge.py*
* Merge the results from the graphical UI elements detection and text recognition 

## How to use?
To test the your own image(s):
* For testing single image, change `PATH_IMG_INPUT` in *main_single*
* For testing a batch of image, change `self.ROOT_INPUT` in *config/CONFIG.py*
* To change the location of the pretrained CNN and CTPN models, revise the `self.MODEL_PATH` and `self.CTPN_PATH` in *config/CONFIG.py*

## Demo
GUI element detection result for web screenshot

![UI Components detection result](https://github.com/MulongXie/UI2CODE/blob/master/demo/uied.png)
