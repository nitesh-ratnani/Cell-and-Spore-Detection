import cv2
import numpy as np
net = cv2.dnn.readNet('yolov3_training_last.weights', 'yolov3_testing.cfg')

classes = []
with open("classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layers_names = net.getLayerNames()
outputlayers= [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

#Loading Image
img = cv2.imread('Image_786.jpg')  #Only change name of image to use
img=cv2.resize(img,None,fx=0.4,fy=0.4)
height, width, channels = img.shape

# Cell/Spore detection
blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

net.setInput(blob)
outs = net.forward(outputlayers)

#Showing info on screen

for out in outs:
    for detection in out:
        scores=detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.4:
            #Object Detected
            center_x = int(detection[0]*width)
            center_y = int(detection[1] * height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            #Rectangle coord
            #x = int(center_x - w/2)
            #y = int(center_y - h/2)
            cv2.circle(img, (center_x, center_y), 10, (0, 255, 0), 2)


cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


