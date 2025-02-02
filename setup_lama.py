import os
import subprocess
import requests
from tqdm import tqdm
import zipfile

def setup_lama():
    # Create lama directory if it doesn't exist
    if not os.path.exists('lama'):
        print("Cloning LaMa repository...")
        subprocess.run(['git', 'clone', 'https://github.com/advimman/lama.git'])
    
    # Download model weights if they don't exist
    if not os.path.exists('lama/big-lama'):
        print("Downloading LaMa weights...")
        url = "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip"
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        # Download zip file
        zip_path = 'lama/big-lama.zip'
        with open(zip_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True
        ) as pbar:
            for data in response.iter_content(1024):
                size = f.write(data)
                pbar.update(size)
        
        # Extract zip file
        print("Extracting weights...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('lama')
        
        # Clean up zip file
        os.remove(zip_path)

if __name__ == "__main__":
    setup_lama() 