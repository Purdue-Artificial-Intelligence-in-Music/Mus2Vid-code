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

class UpscalerThread(threading.Thread):
    
    """
    This function is called when a StableDiffusionThread is created.
    Parameters:
        name: the name of the thread
        prompt: the prompt to use when generating images
        modulus: the amount to increment the seed by when generating images
    Returns: nothing
    """ 
    def __init__(self, 
                 name, 
                 SD_Thread,
                 model_str = 'x2'
                 upsampler = None
                 display_func = None):
        super(UpscalerThread, self).__init__()
        self.SD_Thread = SD_Thread
        self.model_str = model_str
        self.upsampler = upsampler
        if self.upsampler == None:
            self.upsampler = get_upsampler(model_str)
        self.output = None
        self.display_func = display_func
    
    """
    When the thread is started, this function is called which repeatedly generates new Stable Diffusion images.
    Parameters: nothing
    Returns: nothing
    """
    def run(self):
        while self.is_alive():
            image = SD_Thread.pipe.images[0]
            if not image is None:
                img = numpy.array(image)
                sr_image, _ = self.upsampler.enhance(img)
                self.output = [Image.fromarray(sr_image)] 
                if not self.display_func is None:
                    self.display_func(output)