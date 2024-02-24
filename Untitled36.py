#!/usr/bin/env python
# coding: utf-8

# ### CAR & PEDESTRAIN DETECTION 
# pedistrain detection
# In[ ]:


import cv2
import numpy as np

#create our body classifier 
body_classifier = cv2.CascadeClassifier("D:\\Data Science with AI\\object dection\\Haarcascades\\haarcascade_fullbody.xml")

#initiate videio capture for video file
cap = cv2.VideoCapture("C:\\Users\\Achal Raghorte\\OneDrive\\Pictures\\image classification\\walking video 2.mp4")

#loop once video is successfully loaded 
while cap.isOpened():
    
    #read first frame
    ret , frame =cap.read() #'ret':A boolean value indicating whether the frame was successfully read or not. 
    #frame = cv2.resize(frame ,None,fx=0.5, fy=0.5 ,interpolation =cv2.INTER_LINEAR)
    
    gray = cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY)
    #pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray , 1.2, 3)
    
    #extract bounding boxes for any bodies indentified
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame ,(x,y) ,(x+w,y+h),(0,255,255),2)
        cv2.imshow('Pedestrains' , frame)
        
    if cv2.waitKey(1) ==13: #13 is the enter key 
        break
        

cap.release()
cv2.destroyAllWindows()
    


# ### Car Detection

# In[ ]:


import cv2
import time
import numpy as np

#create our body classifier
car_classifier = cv2.CascadeClassifier("D:\\Data Science with AI\\object dection\\Haarcascades\\haarcascade_car.xml")

#initiate video capture for video file
cap = cv2.VideoCapture("D:\\Data Science with AI\\object tracking\\los_angeles.mp4")

#loop once video is successfully loaded
while cap.isOpened():
    
    time.sleep(.05)
    #read first frame 
    ret , frame = cap.read()
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    
    #pass frame to our car classifier 
    cars =car_classifier.detectMultiScale(gray , 1.4, 2)
    
    #extract bounging boxes for any bodies identified 
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x,y) ,(x+w ,y+h) ,(0 ,255,255),2)
        cv2.imshow('Cars' , frame)
    
    if cv2.waitKey(1) == 13: #13 is the enter key
        break
        

cap.release()
cv2.destroyAllWindows()
        


# In[ ]:




