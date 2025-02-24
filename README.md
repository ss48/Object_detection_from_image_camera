Access the page for the TensorFlow Object Detection API: (https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API)

Locate and download the config (.pb) file and the pretrained model weights (.gz) To do this, right-click on the respective links and choose "Save As" to save the files to your computer. (Use existing config file for your model
You can use one of the configs that has been tested in OpenCV. This choice depends on your model and TensorFlow version:)

Find the file called frozen_inference_graph.pb within the folder (.gz). Transfer this file to your working directory, which is the same location where you will create your Python script later in this tutorial.

Additionally, there is a pbtxt file (.pbtxt) obtained from the configuration link. This file is in the PBTXT format.

python3 Object_detection_image.py
python3 Object_detector_camera.py
