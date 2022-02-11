import numpy as np
import cv2

capture = cv2.VideoCapture(0)

#For face and eye detection cv2 comes with pre-trained classifiers which are able to detect features, shapes and patterns in an image 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 
#cv2.data. .... is the path of the haarcascade in the system and the 'haarcascade_frontal....' is the path of the classifier we want to use
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml') 


while True:
    ret,frame = capture.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #COnverting to grayscale image

    #Face Detection
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #(image,scaleFactor,minNeighbors )
    # ScaleFactor - It is used to allow the haarcascade to compare images to be able to detect patterns and shapes...as the classifer is trained accoridng to certain sizes of images
    # The smaller the value the more accurate it is but slower, larger the value the faster it is but less accurate

    # minNeighbors - The no. of neighboring rectangles over an object to detect if its a face or not. If its about 3-5 rectangles then the detection software would be fairly accurate
    
    for(x,y,w,h) in faces: #This is to get the x,y, width and height of an image
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 5) #Drawing the rectangle around the faces it detects
        roi_gray = gray[y:y+w, x:x+w]  #THis is there to take the face out of the picture to detect the eyes...roi - region of interest
        #THis gets the location of the image
        roi_color = frame[y:y+h, x:x+w]  #This is a reference to the colored image

        #Eye Detection
        eyes = eye_cascade.detectMultiScale(roi_gray,1.3,5) #here we are detecting where the eyes on the roi_gray image
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 5)
             #We are using roi_color as we want to draw the image on the colored one as it compares where the eyes are relative to the grayscale image 

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()