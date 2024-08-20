# views.py
import os
import cv2
from django.conf import settings
from django.shortcuts import redirect, render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import base64
import numpy as np
from .models import Images, Video

def snapclick(request):
    return render(request, 'snapclick.html')

def index(request):
    images = Images.objects.all()
    return render(request, 'index.html', {'images': images})

def video (request):
    videos = Video.objects.all()
    return render(request, 'video.html', {'videos': videos})
def video_upscaler(request):
    if request.method == 'POST' and request.FILES.get('video'):
        uploaded_video = request.FILES['video']
        video_name = uploaded_video.name
        video_path = os.path.join(settings.MEDIA_ROOT, 'videos', video_name)
        
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        
        with default_storage.open(video_path, 'wb+') as destination:
            for chunk in uploaded_video.chunks():
                destination.write(chunk)
                
        
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            return render(request, 'index.html', {'error': 'Video could not be read.'})
        
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))

        # Ensure dimensions are divisible by 2
        frame_width = frame_width if frame_width % 2 == 0 else frame_width - 1
        frame_height = frame_height if frame_height % 2 == 0 else frame_height - 1
        
        upscaled_video_path = os.path.join(settings.MEDIA_ROOT, 'videos', 'output.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        video_writer = cv2.VideoWriter(upscaled_video_path, fourcc, fps, (frame_width, frame_height))
        
        video_size = (frame_width, frame_height)
        frame_count_val = frame_count
        fps_val = fps
        print(f"Video name: {video_name}")
        print(f"Video size: {video_size}")
        print(f"Frame count: {frame_count_val}")
        print(f"FPS: {fps_val}")
        for i in range(frame_count):
            ret, frame = video_capture.read()
            if not ret:
                print(f"Failed to read frame {i}")
                break
            upscaled_frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_CUBIC)
            video_writer.write(upscaled_frame)
            print(f"Processed frame {i}/{frame_count}")
        
        video_capture.release()
        video_writer.release()
        
        with open(upscaled_video_path, 'rb') as f:
            upscaled_video_content = ContentFile(f.read(), name='output.mp4')
        upscaled_video_instance = Video.objects.create(
            video=uploaded_video,
            upscaled_video=upscaled_video_content
        )
        upscaled_video_instance.save()
        
        return render(request, 'video.html', {'output_video': upscaled_video_instance.upscaled_video.url , 'videosize' : video_size, 'framecount' : frame_count_val, 'fps' : fps_val})

    return render(request, 'index.html')

def cartoonize(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        image_name = image.name
        image_path = os.path.join(settings.MEDIA_ROOT, 'images', image_name)

        # Make sure the 'images' directory exists
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        # Save the uploaded image
        with default_storage.open(image_path, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)

        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            return render(request, 'index.html', {'error': 'Image could not be read.'})

   


    

        # Create the cartoon effect by combining the grayscale and blurred images
        filter = cv2.bitwise_and(img, img)

        # Convert to HSV color space
        hsv = cv2.cvtColor(filter, cv2.COLOR_BGR2HSV)

        # Define the range of colors to consider saturated
        hsv_lower = np.array([0, 0, 100], dtype=np.uint8)
        hsv_upper = np.array([255, 255, 255], dtype=np.uint8)

        # Apply mask to saturate the colors in the image
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

        # Apply the mask to the image
        filter = cv2.bitwise_and(filter, filter, mask=mask)

        # Add cartoon borders
        width = int(img.shape[1] / 25)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, width))
        filter = cv2.dilate(filter, kernel)
        filter = cv2.erode(filter, kernel)

        # Create the path for the cartoonized image
        filtered_image_dir = os.path.join(settings.MEDIA_ROOT, 'cartoonized')
        os.makedirs(filtered_image_dir, exist_ok=True)
        filtered_image_name = os.path.join(filtered_image_dir, image_name)

        # Save thefiltered image
        cv2.imwrite(filtered_image_name, img)
        cv2.imwrite(filtered_image_name, filter)

        # Savefiltered image in the database
        with open(filtered_image_name, 'rb') as f:
           filtered_image_content = ContentFile(f.read(), name=image_name)

        # Create an instance of the Images model
        filtered_image = Images(
            image=image,
           filtered=filtered_image_content
        )
        filtered_image.save()

        filtered_image_url =filtered_image.filtered.url

        # Encode the image in base64
        with open(filtered_image_name, 'rb') as image_file:
           filtered_image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        return render(request, 'index.html', {'filtered_image':filtered_image_url, 'filtered_image_base64':filtered_image_base64})

    else:
        return redirect('index')


def colorize(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        image_name = image.name
        image_path = os.path.join(settings.MEDIA_ROOT, 'images', image_name)

        # Make sure the 'images' directory exists
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        # Save the uploaded image
        with default_storage.open(image_path, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)

        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            return render(request, 'index.html', {'error': 'Image could not be read.'})

        # Apply color boost
   


        # Boost color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 1] = 255
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img[:, :, 0] = cv2.add(img[:, :, 0], 40)
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        # Saturate image
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.addWeighted(hsv[:, :, 2], 0.7, np.zeros(hsv[:, :, 2].shape, dtype=np.float32), 0, 0)
        v = hsv[:, :, 2].astype(np.float32)
        v = cv2.addWeighted(v, 0.7, np.zeros(v.shape, dtype=np.float32), 0, 0)
        v = np.clip(v, 0, 255).astype(np.uint8)
        hsv[:, :, 2] = v
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Create the cartoon effect
        cartoon = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))

        # Save the filtered image
        # Create the path for the filtered image
        filtered_image_dir = os.path.join(settings.MEDIA_ROOT, 'filtered')
        os.makedirs(filtered_image_dir, exist_ok=True)
        filtered_image_name = os.path.join(filtered_image_dir, image_name)

        # Save the filtered image
        cv2.imwrite(filtered_image_name, img)
        cv2.imwrite(filtered_image_name, cartoon)

        # Save filtered image in the database
        with open(filtered_image_name, 'rb') as f:
            filtered_image_content = ContentFile(f.read(), name=image_name)

        # Create an instance of the Images model
        filtered_image = Images(
            image=image,
            filtered=filtered_image_content
        )
        filtered_image.save()

        filtered_image_url = filtered_image.filtered.url

        # Encode the image in base64
        with open(filtered_image_name, 'rb') as image_file:
            filtered_image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        return render(request, 'index.html', {'filtered_image': filtered_image_url, 'filtered_image_base64': filtered_image_base64})

    else:
        return redirect('index')

def blur(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        image_name = image.name
        image_path = os.path.join(settings.MEDIA_ROOT, 'images', image_name)

        # Make sure the 'images' directory exists
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        # Save the uploaded image
        with default_storage.open(image_path, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)

        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            return render(request, 'index.html', {'error': 'Image could not be read.'})


        # Boost color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 1] = 255
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img[:, :, 0] = cv2.add(img[:, :, 0], 40)
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        # Saturate image
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2].astype(np.float32)
        v = cv2.addWeighted(v, 0.7, np.zeros(v.shape, dtype=np.float32), 0, 0)
        v = np.clip(v, 0, 255).astype(np.uint8)
        hsv[:, :, 2] = v
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # Create the filter effect
        filter = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))

        # Create the path for the cartoonized image
        filtered_image_dir = os.path.join(settings.MEDIA_ROOT, 'filtered')
        os.makedirs(filtered_image_dir, exist_ok=True)
        filtered_image_name = os.path.join(filtered_image_dir, image_name)

        # Save the filtered image
        cv2.imwrite(filtered_image_name, img)
        cv2.imwrite(filtered_image_name, filter)

        # Save filtered image in the database
        with open(filtered_image_name, 'rb') as f:
            filtered_image_content = ContentFile(f.read(), name=image_name)

        # Create an instance of the Images model
        filtered_image = Images(
            image=image,
            filtered=filtered_image_content
        )
        filtered_image.save()

        filtered_image_url = filtered_image.filtered.url

        # Encode the image in base64
        with open(filtered_image_name, 'rb') as image_file:
            filtered_image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        return render(request, 'index.html', {'filtered_image': filtered_image_url, 'filtered_image_base64': filtered_image_base64})

    else:
        return redirect('index')
    
    
def backgroundremover(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        image_name = image_file.name
        image_path = os.path.join(settings.MEDIA_ROOT, 'images', image_name)

        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        with open(image_path, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)

        img = cv2.imread(image_path)

        if img is None:
            return render(request, 'index.html', {'error': 'Image could not be read.'})

        edges = cv2.Canny(img, 100, 200)
        height, width, _ = img.shape
        mask = np.zeros((height + 2, width + 2), np.uint8)
        cv2.floodFill(edges, mask, (0, 0), 255)
        mask = cv2.bitwise_not(mask[1:height + 1, 1:width + 1])
        result = cv2.bitwise_and(img, img, mask=mask)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        dilated_image = cv2.dilate(threshold_image, kernel, iterations=13)

        background = cv2.medianBlur(dilated_image, 21)
        difference = cv2.absdiff(background, dilated_image)
        difference = cv2.cvtColor(difference, cv2.COLOR_GRAY2BGR)

        result = cv2.bitwise_and(result, difference)

        filtered_image_dir = os.path.join(settings.MEDIA_ROOT, 'filtered')
        os.makedirs(filtered_image_dir, exist_ok=True)
        filtered_image_name = os.path.join(filtered_image_dir, image_name)

        cv2.imwrite(filtered_image_name, result)

        filtered_image_content = ContentFile(open(filtered_image_name, 'rb').read(), name=image_name)

        filtered_image = Images(
            image=image_file,
            filtered=filtered_image_content
        )
        filtered_image.save()

        filtered_image_url = filtered_image.filtered.url

        filtered_image_base64 = base64.b64encode(open(filtered_image_name, 'rb').read()).decode('utf-8')

        return render(request, 'index.html', {'filtered_image': filtered_image_url, 'filtered_image_base64': filtered_image_base64})

    return redirect('index')
def blackwhite(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        image_name = image_file.name
        image_path = os.path.join(settings.MEDIA_ROOT, 'images', image_name)

        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        with open(image_path, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)

        img = cv2.imread(image_path)

        if img is None:
            return render(request, 'index.html', {'error': 'Image could not be read.'})

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        filtered_image_dir = os.path.join(settings.MEDIA_ROOT, 'filtered')
        os.makedirs(filtered_image_dir, exist_ok=True)
        filtered_image_name = os.path.join(filtered_image_dir, image_name)

        cv2.imwrite(filtered_image_name, gray_image)

        filtered_image_content = ContentFile(open(filtered_image_name, 'rb').read(), name=image_name)

        filtered_image = Images(
            image=image_file,
            filtered=filtered_image_content
        )
        filtered_image.save()

        filtered_image_url = filtered_image.filtered.url

        filtered_image_base64 = base64.b64encode(open(filtered_image_name, 'rb').read()).decode('utf-8')

        return render(request, 'index.html', {'filtered_image': filtered_image_url, 'filtered_image_base64': filtered_image_base64})
    
def blueish (request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        image_name = image_file.name
        image_path = os.path.join(settings.MEDIA_ROOT, 'images', image_name)
        
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        with open(image_path, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)
                
        img = cv2.imread(image_path)
        
        if img is None:
            return render(request, 'index.html', {'error': 'Image could not be read.'})
        
        filtered_image_dir = os.path.join(settings.MEDIA_ROOT, 'filtered')
        
        os.makedirs(filtered_image_dir, exist_ok=True)
        
        filtered_image_name = os.path.join(filtered_image_dir, image_name)
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(filtered_image_name, img)
        
        filtered_image_content = ContentFile(open(filtered_image_name, 'rb').read(), name=image_name)
        
        filtered_image = Images(
            image=image_file,
            filtered=filtered_image_content
        )
        
        filtered_image.save()
        
        filtered_image_url = filtered_image.filtered.url
        
        filtered_image_base64 = base64.b64encode(open(filtered_image_name, 'rb').read()).decode('utf-8')
        
        return render(request, 'index.html', {'filtered_image': filtered_image_url, 'filtered_image_base64': filtered_image_base64})
        
              
def delete(request, image_id):
    if request.method == 'POST':
        image = Images.objects.get(id=image_id)
        image.delete()

    return redirect('index')

