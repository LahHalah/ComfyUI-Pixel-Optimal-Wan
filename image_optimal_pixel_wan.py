import torch
import numpy as np
from PIL import Image
import math

class imageoptimalpixelwan:
    """
    Scale image to target megapixels with mandatory x16 alignment.
    Optimized for Wan2.2 and other video/image models.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/resize"

    def upscale(self, image, target_megapixels):
        B, H, W, C = image.shape
        aspect_ratio = W / H
        target_total_pixels = target_megapixels * 1_000_000
        
        # Calculate dimensions
        ideal_width = math.sqrt(target_total_pixels * aspect_ratio)
        ideal_height = ideal_width / aspect_ratio
        
        # Force x16 alignment (Rounding)
        new_width = max(16, int(round(ideal_width / 16.0) * 16))
        new_height = max(16, int(round(ideal_height / 16.0) * 16))
        
        print(f"[WanScale] Resizing to: {new_width}x{new_height}")

        output_images = torch.zeros((B, new_height, new_width, C), dtype=torch.float32)
        
        for i in range(B):
            # Safe conversion from any device to CPU Numpy
            img_np = (image[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            
            # Pure Bicubic Resize
            resized_pil = pil_img.resize((new_width, new_height), resample=Image.BICUBIC)
            
            # Back to Tensor
            resized_np = np.array(resized_pil).astype(np.float32) / 255.0
            if len(resized_np.shape) == 2: # Grayscale fix
                resized_np = np.expand_dims(resized_np, axis=-1)
                
            output_images[i] = torch.from_numpy(resized_np)
            del pil_img, resized_pil

        return (output_images,)

NODE_CLASS_MAPPINGS = {"imageoptimalpixelwan": imageoptimalpixelwan}
NODE_DISPLAY_NAME_MAPPINGS = {"imageoptimalpixelwan": "Image Optimal Pixel (Wan)"}
