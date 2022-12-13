from .models import Data,filemodel,statusmodel
from rest_framework import serializers


class DataSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = Data
        fields = ['status','image']

class videoSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = filemodel
        fields = ['lable_status','filefiled']

class statusSerializer(serializers.ModelSerializer):
    class Meta:
        model = statusmodel
        fields = ['status']