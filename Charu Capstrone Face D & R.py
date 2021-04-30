#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
id = input('enter user id')
sampleN=0;
while 1:
    ret, img = cap.read()
#     print(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        sampleN=sampleN+1;
        cv2.imwrite("Project/Dataset."+str(id)+ "." +str(sampleN)+ ".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.waitKey(100)
    cv2.imshow('img',img)
    cv2.waitKey(1)
    if sampleN > 20:
        break

cap.release()
cv2.destroyAllWindows()


# In[4]:


cd/users/Lenovo/Desktop/


# In[5]:


import os
import numpy as np
import cv2
from PIL import Image
# For face recognition we will the the LBPH Face Recognizer 
recognizer = cv2.face.LBPHFaceRecognizer_create();
path="Project/Dataset"
def getImagesWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # print image_path
    #getImagesWithID(path)
    faces = []
    IDs = []
    for imagePath in imagePaths:
        # Read the image and convert to grayscale
        facesImg = Image.open(imagePath).convert('L')
        faceNP = np.array(facesImg, 'uint8')
        # Get the label of the image
        ID= int(os.path.split(imagePath)[-1].split(".")[1])
         # Detect the face in the image
        faces.append(faceNP)
        IDs.append(ID)
        cv2.imshow("Adding faces for traning",faceNP)
        cv2.waitKey(10)
    return np.array(IDs), faces
Ids,faces  = getImagesWithID(path)
recognizer.train(faces,Ids)
recognizer.save("Project/trainer/trainer.xml")
cv2.destroyAllWindows()


# In[6]:


cd/users/Lenovo/Desktop/


# In[6]:


import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create();
rec.read("Project/trainer/trainer.xml")
id=0
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 255, 255)
# font=cv2.InitFont(cv2.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if(id==2):
            id="charu"
        if id==1:
            id="alok"
        if id==3:
            id="anjali"
        if id==4:
            id="Gaurav"
        if id==5:
            id='rahul'
        if id==6:
            id="akshay"
#         cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255)
        cv2.putText(img,str(id),(x,y+h),fontFace,1,(255,255,0),1)
    cv2.imshow('img',img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:




