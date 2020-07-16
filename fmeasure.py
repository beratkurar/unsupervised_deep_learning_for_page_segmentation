# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:14:50 2017

@author: B
"""

import numpy as np
np.random.seed(123)
import glob
import cv2
import os


def fmeasure(truefolder, predfolder):            
    print('calculating the fmeasures')
    epsilon=0.0000000001
    fmain=[]
    fside=[]
    for p in os.listdir(truefolder):
        print(p)
        true=cv2.imread(truefolder+p,0)
        pred=cv2.imread(predfolder+p[:-4]+'.png',1)
        
        rows,cols,ch=pred.shape
        print(pred.shape)
    
        for i in range(rows):
            for j in range(cols):
    
                if pred[i,j,0]==255 and pred[i,j,1]==255 and pred[i,j,2]==255:
                    pass
                if pred[i,j,0]==255 and pred[i,j,1]<255 and pred[i,j,2]<255:
                    pred[i,j,0]=0
                    pred[i,j,1]=0
                    pred[i,j,2]=0
                if pred[i,j,0]<255 and pred[i,j,1]<255 and pred[i,j,2]==255:
                    pred[i,j,0]=128
                    pred[i,j,1]=128
                    pred[i,j,2]=128
        pred[pred<128]=0
        p1=pred>128
        p2=pred<255
        pred[p1*p2]=128
    
        pred=pred[:,:,0]
    
        rows,cols=pred.shape
        print(pred.shape)
        

        mallp=0
        mtp=0
        mfp=0
        mfn=0
        sallp=0
        stp=0
        sfp=0
        sfn=0
        for i in range(rows):
            for j in range(cols):
                if true[i,j]==0:
                    mallp=mallp+1
                if true[i,j]==0 and pred[i,j]==0:
                    mtp=mtp+1
                if true[i,j]==128 and pred[i,j]==0:
                    mfp=mfp+1
                if true[i,j]==0 and pred[i,j]==128:
                    mfn=mfn+1
                if true[i,j]==128:
                    sallp=sallp+1
                if true[i,j]==128 and pred[i,j]==128:
                    stp=stp+1
                if true[i,j]==0 and pred[i,j]==128:
                    sfp=sfp+1
                if true[i,j]==128 and pred[i,j]==0:
                    sfn=sfn+1
        if mallp>0:
            fm=(2.*mtp+epsilon)/(2.*mtp+mfp+mfn+epsilon)
            fmain.append(fm)
            print(p)
            print(fm)
        if sallp>0:
            fs=(2.*stp+epsilon)/(2.*stp+sfp+sfn+epsilon)
            print(p)
            print(fs)
            fside.append(fs)
        print(p+'is finished')
        print('')
    print('main f measure:')
    print(np.mean(fmain))
    print('side f measure:')
    print(np.mean(fside))


truefolder='complex_ltest/'
predfolder='complex_ptest/'
fmeasure(truefolder,predfolder)








