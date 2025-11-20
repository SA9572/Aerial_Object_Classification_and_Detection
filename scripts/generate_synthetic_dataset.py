"""
Generate Synthetic Bird & Drone Images for Testing
===================================================
Creates realistic synthetic images with distinguishing features:
- Birds: Organic shapes, smaller, scattered pixels, wing patterns
- Drones: Geometric shapes, central mass, propeller patterns
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
import random


class SyntheticImageGenerator:
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
    
    def create_bird_image(self):
        """Create synthetic bird image"""
        img = Image.new('RGB', self.img_size, color=(135, 206, 235))  # Sky blue
        draw = ImageDraw.Draw(img, 'RGBA')
        
        # Random position
        cx = random.randint(50, 174)
        cy = random.randint(50, 174)
        
        # Bird body (ellipse-like with wings)
        body_size = random.randint(20, 40)
        
        # Wing pattern
        for i in range(3):
            wing_y = cy + random.randint(-5, 5)
            wing_x_left = cx - body_size - i * 10
            wing_x_right = cx + body_size + i * 10
            draw.line([(wing_x_left, wing_y), (cx - 5, cy)], fill=(40, 20, 10, 200), width=2)
            draw.line([(wing_x_right, wing_y), (cx + 5, cy)], fill=(40, 20, 10, 200), width=2)
        
        # Body
        draw.ellipse(
            [(cx - body_size, cy - body_size//2), (cx + body_size, cy + body_size//2)],
            fill=(60, 40, 20, 220),
            outline=(30, 15, 5, 255),
            width=2
        )
        
        # Head
        head_size = body_size // 3
        draw.ellipse(
            [(cx - head_size, cy - head_size), (cx + head_size, cy)],
            fill=(70, 50, 30, 220),
            outline=(30, 15, 5, 255),
            width=1
        )
        
        # Eye
        draw.ellipse(
            [(cx - head_size//3, cy - head_size//2), (cx - head_size//6, cy - head_size//3)],
            fill=(0, 0, 0, 200)
        )
        
        # Add noise/texture
        img_array = np.array(img)
        noise = np.random.normal(0, 5, img_array.shape).astype(np.uint8)
        img_array = np.clip(img_array.astype(int) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        # Slight blur for realism
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return img
    
    def create_drone_image(self):
        """Create synthetic drone image"""
        img = Image.new('RGB', self.img_size, color=(135, 206, 235))  # Sky blue
        draw = ImageDraw.Draw(img, 'RGBA')
        
        # Random position (more central)
        cx = random.randint(70, 154)
        cy = random.randint(70, 154)
        
        # Drone body (rectangle/square)
        body_size = random.randint(15, 30)
        
        # Central body (darker)
        draw.rectangle(
            [(cx - body_size//2, cy - body_size//2), (cx + body_size//2, cy + body_size//2)],
            fill=(40, 40, 40, 220),
            outline=(20, 20, 20, 255),
            width=2
        )
        
        # Four propellers at corners
        propeller_distance = body_size + 20
        corners = [
            (cx - propeller_distance, cy - propeller_distance),
            (cx + propeller_distance, cy - propeller_distance),
            (cx - propeller_distance, cy + propeller_distance),
            (cx + propeller_distance, cy + propeller_distance),
        ]
        
        for corner_x, corner_y in corners:
            # Propeller arms (cross pattern)
            arm_length = random.randint(15, 25)
            draw.line([(corner_x - arm_length, corner_y), (corner_x + arm_length, corner_y)],
                     fill=(100, 100, 100, 200), width=3)
            draw.line([(corner_x, corner_y - arm_length), (corner_x, corner_y + arm_length)],
                     fill=(100, 100, 100, 200), width=3)
            
            # Propeller circles
            prop_size = random.randint(8, 12)
            draw.ellipse(
                [(corner_x - prop_size, corner_y - prop_size),
                 (corner_x + prop_size, corner_y + prop_size)],
                fill=(150, 150, 150, 180),
                outline=(80, 80, 80, 255),
                width=1
            )
        
        # Arms connecting to body
        for corner_x, corner_y in corners:
            draw.line([(cx, cy), (corner_x, corner_y)], fill=(70, 70, 70, 150), width=2)
        
        # Add noise/texture
        img_array = np.array(img)
        noise = np.random.normal(0, 5, img_array.shape).astype(np.uint8)
        img_array = np.clip(img_array.astype(int) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        # Slight blur for realism
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return img
    
    def generate_dataset(self, output_dir, counts=None):
        """Generate full dataset"""
        if counts is None:
            counts = {
                'train': {'bird': 150, 'drone': 150},
                'valid': {'bird': 30, 'drone': 30},
                'test': {'bird': 30, 'drone': 30}
            }
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        total_generated = 0
        
        for split in ['train', 'valid', 'test']:
            for class_name in ['bird', 'drone']:
                class_dir = output_dir / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                num_images = counts[split][class_name]
                
                for i in range(num_images):
                    if class_name == 'bird':
                        img = self.create_bird_image()
                    else:
                        img = self.create_drone_image()
                    
                    # Save
                    filename = class_dir / f"{class_name}_{split}_{i:04d}.jpg"
                    img.save(filename, quality=85)
                    total_generated += 1
                    
                    if (i + 1) % 50 == 0:
                        print(f"  Generated {i + 1}/{num_images} {class_name} images for {split}")
        
        print(f"\n✓ Dataset generation complete!")
        print(f"  Total images generated: {total_generated}")
        print(f"  Output directory: {output_dir}")
        
        return total_generated


if __name__ == "__main__":
    print("=" * 70)
    print("Synthetic Bird & Drone Dataset Generator")
    print("=" * 70)
    
    # Configuration
    OUTPUT_DIR = Path(__file__).parent.parent / "data" / "classification_dataset"
    
    # Generate
    generator = SyntheticImageGenerator(img_size=(224, 224))
    
    print(f"\nGenerating synthetic images to: {OUTPUT_DIR}\n")
    total = generator.generate_dataset(OUTPUT_DIR)
    
    print(f"\nDataset ready for training!")
    print(f"Expected accuracy with synthetic data: 70-85%")
