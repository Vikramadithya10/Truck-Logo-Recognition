import cv2
import numpy as np

args_image = 'images/5.jpg'
args_config = 'truck_logo_weights/yolov4-obj.cfg'
args_weights = 'truck_logo_weights/yolov4-obj_best.weights'
args_names = 'truck_logo_weights/obj.names'

CONF_THRESH, NMS_THRESH = 0.5, 0.5

# Load the network
net = cv2.dnn.readNetFromDarknet(args_config, args_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get the output layer from YOLO
layers = net.getLayerNames()
output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Read and convert the image to blob and perform forward pass to get the bounding boxes with their confidence scores
img = cv2.imread(args_image)
height, width = img.shape[:2]

blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
layer_outputs = net.forward(output_layers)

class_ids, confidences, b_boxes = [], [], []
for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > CONF_THRESH:
            center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            b_boxes.append([x, y, int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(int(class_id))
            
            

# Perform non maximum suppression for the bounding boxes to filter overlapping and low confident bounding boxes
indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten().tolist()

# Draw the filtered bounding boxes with their class to the image
with open(args_names, "r") as f:
    classes = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

for index in indices:
    x, y, w, h = b_boxes[index]
    cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 1)
    cv2.putText(img, classes[class_ids[index]], (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)

cv2.imshow("image", img)
cv2.imwrite("output/out.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()