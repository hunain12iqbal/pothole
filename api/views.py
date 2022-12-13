from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.response import Response
from .serializers import DataSerializer,videoSerializer,statusSerializer
from .models import Data,filemodel,statusmodel   
from rest_framework import status
from rest_framework.views import APIView
import json
from .detect import detec
from .helpers import modify_input_for_multiple_files
class DataApi(APIView):
    def post(self,request,*args, **kwargs):
        serializer = DataSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            
            res = detec.detect(serializer.data['image']) 
 
            return Response({"status":"pothole","percentage":res}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
    def get(self,request,format=None):
        data = Data.objects.all()
        serializer = DataSerializer(data,many=True)
        return Response(serializer.data)

    


class fileApi(APIView):
    def get(self,request,format=None):
        data = filemodel.objects.all()
        serializer = videoSerializer(data,many=True)
        return Response(serializer.data)

    def post(self,request,format=None):
        serializer = videoSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            
            res = detec.detect(serializer.data['filefiled']) 
            print("res",res)
            return Response({'percentage':res}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class StatusApi(APIView):
    def get(self,request,format=None):
        data =  statusmodel.objects.all()
        serializer = statusSerializer(data,many=True)
        return Response(serializer.data)

    def post(self,request,format=None):
        serializer = statusSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            res = detec.detect(serializer.data['status']) 
            print("res",res)
            return Response({'percentage':res}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
