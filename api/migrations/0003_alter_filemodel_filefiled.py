# Generated by Django 4.0.4 on 2022-10-18 11:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0002_filemodel'),
    ]

    operations = [
        migrations.AlterField(
            model_name='filemodel',
            name='filefiled',
            field=models.FileField(upload_to='video'),
        ),
    ]
