"""
Module for downloading the selected large model.
"""
import os
from huggingface_hub import snapshot_download

def download_model(model_name: str, cache_dir: str = "./model_weights"):
    """
    Downloads the model weights using huggingface_hub.
    Ensure HF_TOKEN is set in your .env or environment variables.
    """
    print(f"Starting download for {model_name}...")
    token = os.environ.get("HF_TOKEN")
    
    if not token:
        print("Warning: HF_TOKEN not found in environment. If this model is gated, the download will fail.")
    
    # Download snapshot
    path = snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        token=token
    )
    
    print(f"Download complete. Model saved to: {path}")
    return path

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Qwen/Qwen2.5-72B-Instruct requires significant disk space (~144GB)
    download_model("Qwen/Qwen2.5-72B-Instruct")
