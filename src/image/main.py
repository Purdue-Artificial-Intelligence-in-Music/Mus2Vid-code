from src.image.StableDiffusion.Diffusion import get_pic
from src.image.Real_ESRGAN.upscale import upscale_image
from src.image.display.display import display_images

if __name__ == "__main__":
    prompt = ""
    img = get_pic(prompt)
    img_upscaled = upscale_image(img[0])
    display_images(img_upscaled)