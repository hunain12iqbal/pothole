from django.contrib import admin
from django.urls import path
from . import views
urlpatterns = [
    
    path('',views.DataApi.as_view(),name='image'),
    path('video/',views.fileApi.as_view()),
    path('status/',views.StatusApi.as_view())

]
