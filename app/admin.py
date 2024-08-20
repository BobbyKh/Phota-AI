from django.contrib import admin

from app.models import Images, Video

# Register your models here.

admin.site.register(Images)
admin.site.register(Video)
admin.site.site_header = "Cartoonize Admin"
admin.site.site_title = "Cartoonize Admin Portal"
admin.site.index_title = "Welcome to Cartoonize Portal"
admin.site.site_url = "https://cartoonize.herokuapp.com/"