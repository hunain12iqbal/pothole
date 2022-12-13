
import torch
import numpy as np
from numpy import random
import cv2
import os
# from yolov5processor.models.experimental import attempt_load
# from yolov5processor.utils.datasets import letterbox
# from yolov5processor.utils.general import (check_img_size, non_max_suppression, scale_coords)
# from yolov5processor.utils.torch_utils import select_device
import colorsys

class InferenceExecutor:
    def __init__(self, weight, confidence=0.4, img_size=640, agnostic_nms=False, gpu=False, iou=0.5, names=[],colors=[]):
        self.weight = weight
        self.confidence = confidence
        self.gpu = gpu
        self.iou = iou
        self.agnostic_nms = agnostic_nms
        self.img_size = img_size
        self.device, self.half = self.inference_device()
        self.classes, self.model, self.names, self.colors = self.load_model(names)
        self.set_colors(colors)
        print("Loaded Models...")

    def inference_device(self):
        if self.gpu:
            device = torch.device('cuda:'+str(torch.cuda.current_device()))
            print("Using GPU Resource(s): {}".format('cuda:'+str(torch.cuda.current_device())))
        else:
            device = torch.device('cpu')
            print("Using CPU Resources")
        half = device.type != 'cpu'
        return device, half

    def load_model(self,names):
        # model = attempt_load(self.weight, map_location=self.device)
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.weight).to(self.device).eval()
        # imgsz = check_img_size(self.img_size, s=model.stride.max())
        # if self.half:
        #     model.half()
        if names==[]:
          names = model.module.names if hasattr(model, 'module') else model.names
        print("Yolo v5 Model Classes: {}".format(names))
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
        # img = torch.zeros((1, 3, imgsz, imgsz), device=self.device)
        # _ = model(img.half() if self.half else img) if self.device.type != 'cpu' else None
        class_map = {index: label for index, label in zip(range(len(names)), names)}
        return class_map, model, names, colors

    # def predict(self, image):
    #     img = letterbox(image, new_shape=self.img_size)[0]
    #     img = img[:, :, ::-1].transpose(2, 0, 1)
    #     img = np.ascontiguousarray(img)
    #     img = torch.from_numpy(img).to(self.device)
    #     img = img.half() if self.half else img.float()
    #     img /= 255.0
    #     if img.ndimension() == 3:
    #         img = img.unsqueeze(0)
    #     pred = self.model(img, augment=False)[0]
    #     pred = non_max_suppression(pred, self.confidence, self.iou, classes=None, agnostic=self.agnostic_nms)
    #     _output = list()
    #     for i, det in enumerate(pred):
    #         if det is not None and len(det):
    #             det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
    #             for *xyxy, conf, cls in reversed(det):
    #                 _output.append({"points": [int(each) for each in xyxy],
    #                                 "conf": float(conf),
    #                                 "class": self.classes[int(cls)]})
    #     return _output
    def predict(self, images):
        results = self.model(images,size=self.img_size)
        formatedResults=[]
        for predictions in results.pandas().xyxy:
            formatedPredictions=[]
            for *xyxy, conf, cls in zip(predictions['xmin'],predictions['ymin'],predictions['xmax'],predictions['ymax'],predictions['confidence'],predictions['class']):
                if float(conf)>self.confidence:
                    formatedPredictions.append({"points": [int(each) for each in xyxy],
                                    "conf": float(conf),
                                    "class": self.classes[int(cls)]})
            formatedResults.append(formatedPredictions)
        return formatedResults



    # def set_colors(self):
    #   """
    #   Generate random colors.
    #   To get visually distinct colors, generate them in HSV space then
    #   convert to RGB.
    #   # """
    #   # brightness = 1.0 if bright else 0.7
    #   # hsv = [(i / N, 1, brightness) for i in range(N)]
    #   # colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    #   # random.shuffle(colors)
    #   # return colors
    #   # brightness = 1.0 
    #   N=len(self.names)
    #   hsv = [(i / N, 1,1) for i in range(N)]
    #   colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    #   self.colors={}
    #   for i, name in enumerate(self.names):
    #     self.colors[name]=colors[i]
    def set_colors(self,colors):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        # """
        if colors==[]:
            N=len(self.names)
            hsv = [( N/(i+1),1,1) for i in range(N)]
            colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        
        self.colors={}
        for i, name in enumerate(self.names):
            self.colors[name]=colors[i]


    def visualize(self,image,predictions,fontScale=0.7,thickness=2):
      N=len(predictions)
      for i, pred in enumerate(predictions):
        box=pred['points']
        color = self.colors[pred['class']]
            
        # Bounding box
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2,y2), (int(color[0]*255),int(color[1]*255),int(color[2]*255)), thickness)
        if pred.get('label',"")!="":
            caption = pred['label']+" : "+pred['class']+" : "+str(pred['conf'])
        else:
            caption=pred['class'] +" : "+str(pred['conf'])
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (x1, y1-6)
        fontColor              = (int(color[0]*255),int(color[1]*255),int(color[2]*255))
        lineType               = 2

        cv2.putText(image,caption, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)



