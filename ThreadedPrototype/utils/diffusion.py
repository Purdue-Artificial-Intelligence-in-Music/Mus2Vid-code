from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
stable_diffusion_model_id = "stabilityai/stable-diffusion-2-1-base"
import torch

DEFAULT_NEGATIVE_PROMPT = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy, low resolution, cropped, beginner, amateur, oversaturated"

def get_pipe():
    pipe = DiffusionPipeline.from_pretrained(stable_diffusion_model_id, torch_dtype=torch.float16, revision="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    return pipe

'''
This class is a thread class that generates images procedurally in real time.
'''

class StableDiffusionThread(threading.Thread):
    
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
                 Prompt_Thread, 
                 inference = 25, 
                 guidance_scale = 7.5
                 imgs_per_prompt = 1):
        super(StableDiffusionThread, self).__init__()
        self.pipe = None
        self.seed = seed
        self.Prompt_Thread = Prompt_Thread
        self.negative_prompt = DEFAULT_NEGATIVE_PROMPT
        self.inference = inference
        self.guidance_scale = guidance_scale
        self.imgs_per_prompt = imgs_per_prompt
        
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")
    
    def set_negative_prompt(self, prompt):
        self.negative_prompt = prompt
    
    """
    When the thread is started, this function is called which repeatedly generates new Stable Diffusion images.
    Parameters: nothing
    Returns: nothing
    """
    def run(self):
        while self.is_alive():
                generator_list = []
                for i in range(num_images_per_prompt):
                    generator_list.append(torch.Generator("cuda"))
                pipe = get_pipe()
                self.pipe = pipe(
                    self.Prompt_Thread.prompt,
                    negative_prompt= self.negative_prompt,
                    num_inference_steps = self.inference,
                    guidance_scale = self.guidance_scale,
                    num_images_per_prompt = self.imgs_per_prompt,
                )