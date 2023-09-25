from realesrgan import RealESRGANer
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
import numpy
from PIL import Image

def get_upsampler(model_str = 'x2'):
    """    
    Downloads the real-ESRGAN model into an upsampler object

    Parameters:
        model_str (str): x2 or x4 default: x2
        
    Returns:
    upsampler object
    """
    if (model_str == 'x2'):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
    elif (model_str == 'x4'):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    model_path = load_file_from_url(url=file_url)
    upsampler = RealESRGANer(scale=netscale,model_path=model_path,model=model,half=True)
    return upsampler

def get_upscaled_image(input, upsampler):
    """
    upscales an input image using an upsampler object

    Parameters:
        input (PIL image): The input image 
        upsampler: upsampler object
    Returns:
        output (list of PIL image): list of upscaled image
    """
    img = numpy.array(input)
    sr_image, _ = upsampler.enhance(img)
    return [Image.fromarray(sr_image)]

def upscale_image(input, model_str = 'x2'):
    """
    downloads a model and upscales the input image

    Parameters:
        input (PIL image): The input image
        model_str (str): x2 or x4 default: x2
    Returns:
        output (list of PIL image): list of upscaled image
    """
    upsampler = get_upsampler(model_str=model_str)
    return get_upscaled_image(input=input, upsampler=upsampler)

if __name__ == "__main__":
    from src.image.display.display import display_images

    filepath = ".png"
    img = Image.open(filepath).convert("RGB")
    img_upscaled = upscale_image(img)
    display_images(img_upscaled)