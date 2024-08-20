from django.conf import settings
from django.contrib import admin
from django.conf.urls.static import static
from . import views
from django.urls import path
urlpatterns = [
    path ('',views.index ,name='index'),
    path ('upload',views.cartoonize ,name='upload'),
    path ('cartoonize',views.cartoonize ,name='cartoonize'),
    path ('colorize',views.colorize ,name='colorize'),
    path ('blur',views.blur ,name='blur'),
    path ('backgroundremover',views.backgroundremover ,name='backgroundremover'),
    path ('blackwhite',views.blackwhite ,name='blackwhite'),
    path ('blueish',views.blueish ,name='blueish'),
    path ('delete/<int:image_id>',views.delete ,name='delete'),
    path ('video',views.video ,name='video'),
    path ('videoupscaler',views.video_upscaler ,name='videoupscaler'),
    path ('snapclick',views.snapclick ,name='snapclick')
    

]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
