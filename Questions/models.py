from django.db import models


# Create your models here.
class TestUPFILE(models.Model):
    name = models.CharField(max_length=600)
    file = models.FileField(upload_to='uploadsT/')
