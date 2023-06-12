from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
stable_diffusion_model_id = "stabilityai/stable-diffusion-2-1-base"
import torch

def get_pipe():
    pipe = DiffusionPipeline.from_pretrained(stable_diffusion_model_id, torch_dtype=torch.float16, revision="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    return pipe

def get_pic(prompt,negative_prompt = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"
            , inference = 10, guidance_scale = 7.5, num_images_per_prompt = 1):
    """
    Creates an image using stable diffusion pipeline
    Parameters:
        prompt: string of the prompt
        negative prompt: string of the negative prompt
        inference: number of inference step, around 50 for a high quality image
        guidance scale: a way to increase the adherence to the conditional signal that guides the generation (text, in this case) as well as overall sample quality
    Returns:
        output: stable diffusion pipeline output
    """
    generator_list = []
    for i in range(num_images_per_prompt):
        generator_list.append(torch.Generator("cuda"))
    pipe = get_pipe()
    return pipe(
        prompt,
        negative_prompt= negative_prompt,
        num_inference_steps = inference,
        guidance_scale = guidance_scale,
        num_images_per_prompt = num_images_per_prompt,
    )

def display_images(pipe):
    for i in range(len(pipe[0])):
        image = pipe.images[i]
        image.show()