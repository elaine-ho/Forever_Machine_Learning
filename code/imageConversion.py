import cv2
from PIL import Image

# Crops image file to a square with a centered pivot then resizes to width, height
def crop_and_resize(filename, width, height)
    img = cv2.imread(filename)
    if img.shape[1] > img.shape[0]:
        start_y = int(img.shape[1]/2 - img.shape[0]/2)
        end_y = start_y + img.shape[0]
        crop = img[0:img.shape[0], start_y:end_y]
    else: 
        start_x = int(img.shape[0]/2 - img.shape[1]/2)
        end_x = start_x + img.shape[1]
        crop = img[start_x:end_x, 0:img.shape[1]]
    return cv2.resize(crop, (width, height))

# TO USE: img.getpixel((x,y)) for rgb at x,y
def rgb(cv2_image):
    cvtcolor = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cvtcolor)
    return img.convert("RGB")