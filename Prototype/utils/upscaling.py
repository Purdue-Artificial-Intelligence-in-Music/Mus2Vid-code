from realesrgan import RealESRGANer
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

def get_upscaler(model_str = 'x2'):
    """    
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

def get_upscaled_image(input, output, upsampler):
    """
    Parameters:
        input (str): The path to the input image file
        output (str): The path to the output image file
        upsampler: upsampler object
    """
    img = cv2.imread(input)
    sr_image, _ = upsampler.enhance(img)
    cv2.imwrite(output,sr_image)

def upscale_image(input, output, model_str = 'x2'):
    """
    Parameters:
        input (str): The path to the input image file
        output (str): The path to the output image file
        model_str (str): x2 or x4 default: x2
    """
    upscaler = get_upscaler(model_str=model_str)
    get_upscaled_image(input=input, output=output, upsampler=upscaler)