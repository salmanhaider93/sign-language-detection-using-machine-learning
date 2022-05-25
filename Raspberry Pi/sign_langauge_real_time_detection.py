import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

## path for testing cofing file and tained model form colab
net = cv2.dnn.readNetFromDarknet("yolov3_custom.cfg","yolov3_custom_last.weights")

### classes for trained model 

classes = ['Good Job','Hello','I Love You','No','Yes','Thank You',]

x =  480#640
y = 320#480
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (x, y)
camera.framerate = 20
rawCapture = PiRGBArray(camera, size=(x, y))

# allow the camera to warmup
time.sleep(0.1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    img = frame.array

    #img = cv2.resize(img,(160,120))
    hight,width,_ = img.shape
    blob = cv2.dnn.blobFromImage(img,1/255,size=(256,256),mean=(0,0,0),swapRB = True, crop = False)
    #blob = cv2.dnn.blobFromImage(img, swapRB=True, crop=False)
    net.setInput(blob)

    output_layers_name = net.getUnconnectedOutLayersNames()

    layerOutputs = net.forward(output_layers_name)

    boxes =[]
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.1:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3]* hight)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                print(confidences)
                print(class_id)
                print(classes[class_id])
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes,confidences,.3,.2)

    boxes =[]
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.1:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3]* hight)

                x = int(center_x - w/2)
                y = int(center_y - h/2)



                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes,confidences,.3,.2)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255,size =(len(boxes),3))
    if  len(indexes)>0:
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            cv2.putText(img,label + " " + confidence, (x,y),font,2,color,2)

    cv2.imshow('img',img)
    if cv2.waitKey(1) == ord('q'):
        break

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    
cv2.destroyAllWindows()