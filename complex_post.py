import cv2
import os
import numpy as np

maintextfolder='complex_maintext_4'
predfolder='complex_ptest'
labelfolder='complex_ltest'

for imgname in os.listdir(maintextfolder):
    mask=cv2.imread(os.path.join(maintextfolder,imgname),0)
    imglabelname=imgname[:-4]+'.bmp'
    label=cv2.imread(os.path.join(labelfolder,imglabelname),0)

    # get the contours
    ret,thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # find the biggest countour by area
    c = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    x=x+30
    y=y+30
    w=w-30
    h=h-30

    # draw the biggest contour
    #cv2.rectangle(label,(x,y),(x+w,y+h),(0,255,0),2)

    # save the label
    #cv2.imwrite(os.path.join(predfolder,imgname),label)

    pred=label.copy()
    #get the maintext components
    mask=np.zeros(mask.shape)
    mask[y:y+h,x:x+w]=255
    m=(mask==255) & (pred<255)
    pred[m]=0
    #get the sidetext components
    mask=np.zeros(mask.shape)
    mask[:,0:x]=255
    mask[:,x+w:]=255
    mask[:y,:]=255
    mask[y+h:,:]=255
    s=(mask==255) & (pred<255)
    pred[s]=128

    # save the images
    cv2.imwrite(os.path.join(predfolder,imgname),pred)
    

    
