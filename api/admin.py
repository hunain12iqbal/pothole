from django.contrib import admin
from .models import Data,filemodel

# Register your models here.

@admin.register(Data)
class StudentData(admin.ModelAdmin):
    list_display = ['status','image']
    
@admin.register(filemodel)
class filedmodel(admin.ModelAdmin):
    list_display = ['lable_status','filefiled']