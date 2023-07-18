def display_images(images):
    """
    Displays images in a StableDiffusionPipelineOutput
    Parameters:
        images: StableDiffusionPipelineOutput
    Returns:
        void
    """
    for i in range(len(images)):
        image = images[i]
        image.show()