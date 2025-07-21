import os
import torch
from visual_anagrams.views import get_views
from visual_anagrams.samplers import get_sampler
import pickle
from pathlib import Path

def generate_illusion():
    # Create output directory
    output_dir = Path('results/rotate_cw.village.horse')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize view and sampler
    views = get_views(['rotate_cw'])
    view = views[0]  # Get the first (and only) view
    sampler = get_sampler('village.horse')
    
    # Generate samples
    samples = sampler.sample(num_samples=3, sizes=[64, 256, 1024])
    
    # Save samples and metadata
    metadata = {
        'view_type': 'rotate_cw',
        'prompt': 'village.horse',
        'samples': samples
    }
    
    # Save metadata
    with open(output_dir / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save images
    for size, sample in zip([64, 256, 1024], samples):
        sample.save(output_dir / f'sample_{size}.png')
    
    print(f"Illusion generated and saved to {output_dir}")

if __name__ == "__main__":
    generate_illusion() 