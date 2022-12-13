from django.db import models

# Create your models here.
class Data(models.Model):

    status = models.CharField(max_length=50)
    image = models.ImageField(upload_to="my_picture",blank=True)

    def __str__(self):
        return self.status

class filemodel(models.Model):
    lable_status = models.CharField(max_length=50)
    filefiled = models.FileField(upload_to="video", max_length=100)
    def __str__(self):
        return self.lable_status


class statusmodel(models.Model):
    status = models.CharField(max_length=50)
    def __str__(self):
        return self.status