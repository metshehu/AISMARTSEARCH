from django.db import models


# Create your models here.
class History(models.Model):
    # The user who sent the question (sender)
    sender = models.CharField(max_length=255)

    # The question that was asked
    question = models.TextField()

    # The answer given to the question
    respons = models.TextField()
