import os
import platform
import shutil
import subprocess

def get_hw_info():
    # CPU
    cpu_name = platform.processor()
    cpu_cores = os.cpu_count()
    
    # RAM
    print("Total RAM: ~16GB (from systeminfo)")
    
    # Disk
    total, used, free = shutil.disk_usage("/")
    print(f"Free Disk Space: {free // (2**30)} GB")
    
    # GPU
    try:
        # Use python subprocess to run powershell command
        gpu_info = subprocess.check_output(["powershell", "-Command", "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM | ConvertTo-Json"]).decode('utf-8')
        print(f"GPU Info: {gpu_info}")
    except Exception as e:
        print(f"No GPU found or error: {e}")

if __name__ == "__main__":
    get_hw_info()
