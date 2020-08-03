# UIED - UI element detection part of UI2CODE, detecting UI elements from UI screenshots or drawnings

## What is it?

UI Element Detection (UIED) is an old-fashioned computer vision (CV) based element detection approach for graphic user interface. 

The input of UIED could be various UI image, such as mobile app or web page screenshot, UI design drawn by Photoshop or Sketch, and even some hand-drawn UI design. Then the approach detects and classifies text and graphic UI elements, and exports the detection result as JSON file for future application. 

UIED comprises two parts to detect UI text and graphic elements, such as button, image and input bar. 
* For text, it leverages a state-of-the-art scene text detector [EAST](https://github.com/argman/EAST) to perfrom detection. 

* For graphical elements, it uses old-fashioned CV and image processing algorithms with a set of creative innovations to locate the elements and applies a CNN to achieve classification. 
   
## File structure
*config/*
* Set path of the CNN training result (UI components classification) and OCR training result (text recognition) 
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
