import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import torch.nn.functional as F
import gdown  # For downloading from Google Drive
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm  # For better progress bars
import requests
from pathlib import Path
import subprocess

class E2FGVI_HQ(nn.Module):
    def __init__(self):
        super().__init__()
        # Enhanced architecture for HQ model
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(True)
        )
        
        self.middle = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, mask):
        # Enhanced forward pass
        masked_input = x * (1 - mask)
        features = self.encoder(masked_input)
        features = self.middle(features)
        output = self.decoder(features)
        comp = output * mask + x * (1 - mask)
        return comp

def download_model_weights():
    weights_dir = "weights"
    weights_path = os.path.join(weights_dir, "e2fgvi_hq.pth")
    
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    
    if not os.path.exists(weights_path):
        print("Downloading model weights...")
        url = "https://huggingface.co/spaces/VIPLab/Track-Anything/resolve/main/checkpoints/E2FGVI-HQ-CVPR22.pth"
        
        # Download with progress bar
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(weights_path, 'wb') as f, tqdm(
            desc="Downloading weights",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                pbar.update(size)
                
        print("Download complete!")
    
    return weights_path

def download_lama_model():
    model_url = "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.pt"
    model_dir = "weights"
    model_path = os.path.join(model_dir, "big-lama.pt")
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    if not os.path.exists(model_path):
        print("Downloading LaMa model...")
        response = requests.get(model_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True
        ) as pbar:
            for data in response.iter_content(1024):
                size = f.write(data)
                pbar.update(size)
    
    return model_path

def process_video_gan(video_path, window_size=5):
    device = torch.device('cpu')
    # Update model initialization
    model = E2FGVI_HQ().to(device)
    
    weights_path = download_model_weights()
    state_dict = torch.load(weights_path, map_location=device)
    # Remove module prefix if present in state dict
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Read video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup output
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    
    output_path = os.path.join(outputs_dir, f'{base_name}_inpainted_gan.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Open mask video
    mask_path = os.path.join(outputs_dir, f'{base_name}_mask_only.mp4')
    mask_cap = cv2.VideoCapture(mask_path)

    # Buffer for temporal consistency
    frame_buffer = []
    mask_buffer = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        mask_ret, mask = mask_cap.read()
        
        if not ret or not mask_ret:
            break

        frame_count += 1
        print(f'\rProcessing frame {frame_count}/{total_frames} (GAN)', end='')

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) / 255.0

        frame_buffer.append(frame_rgb)
        mask_buffer.append(mask_gray)

        if len(frame_buffer) >= window_size:
            frames_tensor = torch.FloatTensor(np.stack(frame_buffer)).permute(0, 3, 1, 2)
            masks_tensor = torch.FloatTensor(np.stack(mask_buffer)).unsqueeze(1)

            with torch.no_grad():
                output = model(frames_tensor, masks_tensor)

            middle_idx = window_size // 2
            output_frame = output[middle_idx].permute(1, 2, 0).numpy()
            output_bgr = (output_frame * 255).astype(np.uint8)
            output_bgr = cv2.cvtColor(output_bgr, cv2.COLOR_RGB2BGR)
            out.write(output_bgr)

            frame_buffer.pop(0)
            mask_buffer.pop(0)

    # Process remaining frames
    while frame_buffer:
        frames_tensor = torch.FloatTensor(np.stack(frame_buffer)).permute(0, 3, 1, 2)
        masks_tensor = torch.FloatTensor(np.stack(mask_buffer)).unsqueeze(1)

        with torch.no_grad():
            output = model(frames_tensor, masks_tensor)

        output_frame = output[0].permute(1, 2, 0).numpy()
        output_bgr = (output_frame * 255).astype(np.uint8)
        output_bgr = cv2.cvtColor(output_bgr, cv2.COLOR_RGB2BGR)
        out.write(output_bgr)

        frame_buffer.pop(0)
        mask_buffer.pop(0)

    cap.release()
    mask_cap.release()
    out.release()
    print(f"\nGAN Processing complete. Output saved to: {output_path}")

def process_video_lama(video_path, window_size=5):
    # Ensure PYTHONPATH and TORCH_HOME are set
    os.environ['PYTHONPATH'] = os.path.join(os.getcwd(), 'lama')
    os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), 'lama')
    
    # Create temporary directories for frames and masks
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    mask_dir = "temp_masks"
    os.makedirs(mask_dir, exist_ok=True)
    output_dir = "temp_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract frames and masks
    cap = cv2.VideoCapture(video_path)
    mask_path = os.path.join("outputs", f'{os.path.splitext(os.path.basename(video_path))[0]}_mask_only.mp4')
    mask_cap = cv2.VideoCapture(mask_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print("Extracting frames and masks...")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        mask_ret, mask = mask_cap.read()
        if not ret or not mask_ret:
            break
            
        frame_path = os.path.join(temp_dir, f"frame_{frame_count:06d}.png")
        mask_path = os.path.join(mask_dir, f"frame_{frame_count:06d}.png")
        
        cv2.imwrite(frame_path, frame)
        cv2.imwrite(mask_path, mask)
        frame_count += 1
    
    cap.release()
    mask_cap.release()
    
    # Run LaMa prediction
    print("Running LaMa inpainting...")
    lama_command = [
        'python3',
        'lama/bin/predict.py',
        f'model.path={os.path.join(os.getcwd(), "lama/big-lama")}',
        f'indir={os.path.join(os.getcwd(), temp_dir)}',
        f'outdir={os.path.join(os.getcwd(), output_dir)}'
    ]
    subprocess.run(lama_command)
    
    # Create output video
    output_path = os.path.join("outputs", f'{os.path.splitext(os.path.basename(video_path))[0]}_inpainted_lama.mp4')
    
    # Get first frame to determine size
    first_output = cv2.imread(os.path.join(output_dir, "frame_000000.png"))
    height, width = first_output.shape[:2]
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    print("Creating output video...")
    for i in range(frame_count):
        frame_path = os.path.join(output_dir, f"frame_{i:06d}.png")
        frame = cv2.imread(frame_path)
        out.write(frame)
    
    out.release()
    
    # Cleanup temporary files
    print("Cleaning up...")
    import shutil
    shutil.rmtree(temp_dir)
    shutil.rmtree(mask_dir)
    shutil.rmtree(output_dir)
    
    print(f"Processing complete. Output saved to: {output_path}")

def process_video_diffusion(video_path, prompt="Fire", max_size=256):
    # Force CPU usage
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cpu')
    
    # Initialize the pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float32,
        device=device
    )
    
    #pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    pipe = pipe.to(device)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate resize ratio
    ratio = min(max_size / frame_width, max_size / frame_height)
    new_width = int(frame_width * ratio)
    new_height = int(frame_height * ratio)
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    
    output_path = os.path.join(outputs_dir, f'{base_name}_inpainted_diffusion.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    mask_path = os.path.join(outputs_dir, f'{base_name}_mask_only.mp4')
    mask_cap = cv2.VideoCapture(mask_path)

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        mask_success, mask = mask_cap.read()
        
        if not success or not mask_success:
            break

        frame_count += 1
        print(f'\rProcessing frame {frame_count}/{total_frames} (Diffusion)', end='')

        # Resize for processing
        frame_small = cv2.resize(frame, (new_width, new_height))
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_small = cv2.resize(mask_gray, (new_width, new_height))

        # Convert to PIL Images
        image_pil = Image.fromarray(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask_small)

        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                image=image_pil,
                mask_image=mask_pil,
                num_inference_steps=10,
                guidance_scale=7.0,
                height=new_height,
                width=new_width
            ).images[0]

        output_np = np.array(output)
        output_full = cv2.resize(output_np, (frame_width, frame_height))
        
        mask_3ch = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR) / 255.0
        output_bgr = cv2.cvtColor(output_full, cv2.COLOR_RGB2BGR)
        final_frame = (output_bgr * mask_3ch + frame * (1 - mask_3ch)).astype(np.uint8)

        out.write(final_frame)

    cap.release()
    mask_cap.release()
    out.release()
    print(f"\nDiffusion Processing complete. Output saved to: {output_path}")

if __name__ == "__main__":
    videos_dir = "videos"
    if not os.path.exists(videos_dir):
        os.makedirs(videos_dir)
        
    for video_file in os.listdir(videos_dir):
        if video_file.lower().endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(videos_dir, video_file)
            print(f"Processing: {video_path}")
            
            # Choose which method to use (or use both)
           # process_video_gan(video_path)
            #process_video_lama(video_path)           
            process_video_diffusion(video_path)  # Uncomment to use diffusion
