import cv2 as cv
import numpy as np

# Class labels that the model is trained to detect
classes = ["background", "person", "bicycle", "car", "motorcycle",
  "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
  "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
  "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
  "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
  "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
  "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
  "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
  "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
  "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
  "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
  "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# Colors for the object labels
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Path to the TensorFlow model and config
pb = 'frozen_inference_graph.pb'
pbt = 'ssd_inception_v2_coco_2017_11_17.pbtxt'

# Load the neural network
cvNet = cv.dnn.readNetFromTensorflow(pb, pbt)

def process_image(image_path):
    # Load the image
    img = cv.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    rows = img.shape[0]
    cols = img.shape[1]
    cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))

    # Run object detection
    cvOut = cvNet.forward()

    # Go through each object detected and label it
    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        if score > 0.3:
            idx = int(detection[1])
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
                
            label = "{}: {:.2f}%".format(classes[idx], score * 100)
            y = top - 15 if top - 15 > 15 else top + 15
            cv.putText(img, label, (int(left), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

    # Display the image
    cv.imshow('Detected Objects', img)
    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

# Example of processing an image
process_image('12345.jpeg')

