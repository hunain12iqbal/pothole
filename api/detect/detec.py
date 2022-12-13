import cv2
import os
from matplotlib import pyplot as plt
from . import InferenceExecutor

from .tracker import *
import csv  
import cv2
import requests
import json


def detect(input):
    vid = r"../video"
    s_list = input.split("/")
    imgs = ""

    inputVideoPath = ""
    for i in s_list:
        print(i)
        if i == "video":
            
            inputVideoPath = f"D:/HP/Documents/pothole/data/detect/project/pothole_api/pothhole_api/media/video/{s_list[len(s_list)-1]}"
            # inputVideoPath = imgs
            
        elif input == str(0):
            inputVideoPath = int(input)
        elif i == "my_picture": 
            inputVideoPath = "http://127.0.0.1:8000"+input
        else:
            pass
    
    # imgs = kk+"video.mkv"
    ss =  r"api/best.pt"
    model = InferenceExecutor.InferenceExecutor(weight=ss, confidence=0.25, \
                img_size=640, agnostic_nms=False, gpu=False, iou=0.5,names=['pothole'],colors=[(255,0,0)])






    ShowOutputVideo=True
    # WriteOutputVideo=True
    objectTypesToKeep=['pothole']


    cap = cv2.VideoCapture(inputVideoPath)

    li = []
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        predictions = model.predict([frame])
        print("prediction",predictions)
        
        
        if predictions != [[]]:
            model.visualize(frame,predictions[0],fontScale=0.7,thickness=1)
            for i in predictions: 
                li.append(i) 

        

        cv2.imshow('Frame', frame)
            
        #     # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        # if WriteOutputVideo:
        #     out.write(frame)
        

        # Closes all the frames
    cv2.destroyAllWindows()
    cap.release()
        # out.release()
    conf = 0
    c = []
    for i in li:
        for j in i:
            conf = conf + j['conf'] 
            c.append(j['conf'])
    cons = 0
    try:
        cons = (conf / len(c)) * 100
    except:
        pass


    return cons