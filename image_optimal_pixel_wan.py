import torch
import numpy as np
from PIL import Image
import math
import comfy.utils

class imageoptimalpixelwan:
    """
    Expert-level node to scale images to a target megapixel count 
    while maintaining a strict x16 alignment for width and height.
    Mandatory: Bicubic interpolation, Batch support, and x16 Safe.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_megapixels": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.01, 
                    "max": 16.0, 
                    "step": 0.01,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/resize"

    def upscale(self, image, target_megapixels):
        # Image shape: [Batch, Height, Width, Channels]
        B, H, W, C = image.shape
        
        # 1. Hitung Aspect Ratio asli (W/H)
        aspect_ratio = W / H
        
        # 2. Hitung target total pixels (1 MP = 1,000,000 pixels)
        target_total_pixels = target_megapixels * 1_000_000
        
        # 3. Hitung dimensi ideal (floating point)
        # Formula: W = sqrt(Total_Pixels * Aspect_Ratio)
        ideal_width = math.sqrt(target_total_pixels * aspect_ratio)
        ideal_height = ideal_width / aspect_ratio
        
        # 4. Round ke kelipatan 16 terdekat (x16 Safe Alignment)
        # Wajib agar kompatibel dengan model Wan2.2/Video VAE
        new_width = max(16, int(round(ideal_width / 16.0) * 16))
        new_height = max(16, int(round(ideal_height / 16.0) * 16))
        
        # Log hasil kalkulasi ke konsol
        print(f"[imageoptimalpixelwan] Target: {new_width}x{new_height} (Approx {target_megapixels}MP)")

        # Pre-allocate output tensor di CPU untuk stabilitas memori
        output_images = torch.zeros((B, new_height, new_width, C), dtype=torch.float32)
        
        # 5. Loop setiap gambar dalam batch
        for i in range(B):
            # Ambil satu gambar, pastikan berada di CPU sebelum konversi numpy
            img_single = image[i].cpu().numpy()
            
            # Convert ke PIL (Format 0-255 uint8)
            img_pil = Image.fromarray(np.clip(img_single * 255.0, 0, 255).astype(np.uint8))
            
            # 6. Resize dengan BICUBIC (sesuai spesifikasi instruksi)
            resized_pil = img_pil.resize((new_width, new_height), resample=Image.BICUBIC)
            
            # 7. Kembalikan ke format ComfyUI (Float32 Tensor 0-1)
            resized_np = np.array(resized_pil).astype(np.float32) / 255.0
            
            # Penanganan kasus khusus jika channel hilang saat konversi
            if len(resized_np.shape) == 2: # Jika image menjadi grayscale
                resized_np = np.expand_dims(resized_np, axis=-1)
                
            output_images[i] = torch.from_numpy(resized_np)
            
            # Pembersihan objek lokal per iterasi untuk manajemen memori
            del img_pil, resized_pil

        # Kembalikan Batch Tensor hasil resize
        return (output_images,)

# Registrasi Class untuk ComfyUI
NODE_CLASS_MAPPINGS = {
    "imageoptimalpixelwan": imageoptimalpixelwan
}

# Registrasi Nama Tampilan di UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "imageoptimalpixelwan": "Image Scale To Total Pixels (x16 Safe - Bicubic)"
}
