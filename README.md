# UIED - UI element detection part of UI2CODE, detecting UI elements from UI screenshots or drawnings

>This project is still ongoing and this repo may be updated irregularly, I also implement a web app for this project in http://uied.online

## Related Publications: 
[1. UIED: a hybrid tool for GUI element detection](https://dl.acm.org/doi/10.1145/3368089.3417940)

[2. Object Detection for Graphical User Interface: Old Fashioned or Deep Learning or a Combination?](https://arxiv.org/abs/2008.05132)

## What is it?

UI Element Detection (UIED) is an old-fashioned computer vision (CV) based element detection approach for graphic user interface. 

The input of UIED could be various UI image, such as mobile app or web page screenshot, UI design drawn by Photoshop or Sketch, and even some hand-drawn UI design. Then the approach detects and classifies text and graphic UI elements, and exports the detection result as JSON file for future application. 

UIED comprises two parts to detect UI text and graphic elements, such as button, image and input bar. 
* For text, it leverages a state-of-the-art scene text detector [EAST](https://github.com/argman/EAST) to perfrom detection. 

* For graphical elements, it uses old-fashioned CV and image processing algorithms with a set of creative innovations to locate the elements and applies a CNN to achieve classification. 
   
## How to use?

### Dependency
* **Python 3.5**
* **Numpy 1.15.2**
* **Opencv 3.4.2**
* **Tensorflow 1.10.0**
* **Keras 2.2.4**
* **Sklearn 0.22.2**
* **Pandas 0.23.4**

### Installation
Install the mentioned dependencies, and download two pre-trained models from [this link](https://drive.google.com/drive/folders/1MK0Om7Lx0wRXGDfNcyj21B0FL1T461v5?usp=sharing) for EAST text detection and GUI element classification.

Change ``CNN_PATH`` and ``EAST_PATH`` in *config/CONFIG.py* to your locations.

### Usage
To test your own image(s):
* For testing single image, change ``input_path_img`` in *run_single.py* to your own input image and the results will be outputted to ``output_root``.
* For testing mutiple images, change ``input_img_root`` in *run_batch.py* to your own input directory and the results will be outputted to ``output_root``.

> Note: The best set of parameters vary for different types of GUI image (Mobile App, Web, PC). Three of critical ones are ``{'param-grad', 'param-block', 'param-minarea'}`` which can be easily adjusted in *detect_compo\ip_region_proposal.py*.
   
## File structure
*cnn/*
* Used to train classifier for graphic UI elements
* Set path of the CNN classification model

*config/*
* Set data paths 
* Set parameters for graphic elements detection

*data/*
* Input UI images and output detection results

*detect_compo/*
* Graphic UI elemnts localization
* Graphic UI elemnts classification by CNN

*detect_text_east/*
* UI text detection by EAST

*result_processing/*
* Result evaluation and visualizition

*merge.py*
* Merge the results from the graphical UI elements detection and text detection 

*run_batch.py*
* Process a batch of images 

*run_single.py*
* Process a signle image


## Demo
GUI element detection result for web screenshot
 
![UI Components detection result](https://github.com/MulongXie/UIED/blob/master/data/demo/demo.png)
