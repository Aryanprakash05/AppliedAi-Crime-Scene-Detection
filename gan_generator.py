import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import matplotlib.pyplot as plt
import os

class CrimeSceneGenerator:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,  
            safety_checker=None,        
            requires_safety_checker=False
        ).to(self.device)
        
        if self.device == "mps":
            self.pipe.enable_attention_slicing()
            torch.mps.empty_cache()
        
    def generate_scene(self, text, output_path="data/generated/crime_scene.png"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        crime_prompt = (
            "policde investigation, crime scene, "
            "forensic evidence, blood stains, "
            "evidence markers, scattered objects, "
            "forensic crime scene photo, high detail, 4k, "
            "realistic lighting, police investigation, "
            f"{text[:150]}"
        )
        
        negative_prompt = (
            "cartoon, 3d, drawing, painting, sketch, "
            "blurry, low quality, unrealistic"
        )
        
        try:
            with torch.no_grad():
                image = self.pipe(
                    prompt=crime_prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=9.0,
                    num_inference_steps=30,
                    height=512,
                    width=512
                ).images[0]
            
            image.save(output_path)
            self._display_image(image)
            return output_path
            
        except RuntimeError as e:
            print(f"⚠️ Generation failed: {str(e)}")
            print("🔄 Trying fallback CPU method...")
            return self._fallback_generation(crime_prompt, negative_prompt, output_path)
    
    def _fallback_generation(self, prompt, negative_prompt, output_path):
        """Fallback to CPU if MPS fails"""
        self.pipe = self.pipe.to("cpu")
        with torch.no_grad():
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=7.5,
                num_inference_steps=25
            ).images[0]
        
        image.save(output_path)
        self._display_image(image)
        return output_path
    
    def _display_image(self, image):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.title("Generated Crime Scene")
        plt.tight_layout()
        plt.show()

def test_generator():
    generator = CrimeSceneGenerator()
    test_text = (
        "A man found dead in an art gallery, "
        "blood stains on chest, scattered art supplies, "
        "police tape, forensic investigators"
    )
    generated_path = generator.generate_scene(test_text)
    print(f"Generated image saved to: {generated_path}")

if __name__ == "__main__":
    test_generator()