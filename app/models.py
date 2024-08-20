from django.db import models

# Create your models here.


class Images(models.Model):
    image = models.ImageField(upload_to='images/')
    filtered = models.ImageField(upload_to='filtered/')
  

    def __str__(self):
        return self.image.url
    

class Video (models.Model):
    video = models.FileField(upload_to='videos/')
    upscaled_video = models.FileField(upload_to='upscaled_videos/')

    def __str__(self):
        return self.video.url
    