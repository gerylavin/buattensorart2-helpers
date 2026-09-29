from huggingface_hub import snapshot_download,hf_hub_download
import os
from tqdm.auto import tqdm
import subprocess
import shutil


CKPT_PATH="/root/checkpoint"

def snapshot(repo_id,ckpt_pth=CKPT_PATH,ignore=[],allow_patterns=["*"]):
    
    
    os.makedirs(ckpt_pth,exist_ok=True)
    
    try:
        ckpt = snapshot_download(
            repo_id=repo_id,
            cache_dir="/cache",
            ignore_patterns=ignore,
            allow_patterns=allow_patterns,
            max_workers=32,
            tqdm_class=tqdm
            
        )
    except Exception as e:
        print(e)
        print(f"failed to snapshot_download {repo_id}")
    
    
    try:
        subprocess.run(
            f"ln -s {ckpt}/* {ckpt_pth}",
            shell=True,
            check=True,
        )
    except Exception as e:
        print(e)
        print(f"failed to make symlinks for {repo_id}")
    
    
def hf_hub(repo_id,filename,ckpt_pth=CKPT_PATH):
 
    
    try:
        ckpt = hf_hub_download(
            repo_id=repo_id,
            cache_dir="/cache",
            repo_type="model",
            filename=filename,
            tqdm_class=tqdm,
        
        )
    except Exception as e:
        print(e)
        print(f"failed to hf_hub_download {repo_id}")
    
    try:
        subprocess.run(
            f"ln -s {ckpt} {ckpt_pth}/{filename}",
            shell=True,
            check=True,
        )
    except Exception as e:
        print(e)
        print(f"failed to make symlinks for {repo_id}")
        
        
def getdatasethf(
    mydatasets:list,
    lora_name:str,
    repo_id:str
):
    
    allow_patterns=[f"{lora_name}/{i.dataset_name}/*" for i in mydatasets]
    
    try:
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            local_dir="/root"
        )
        
        for i in mydatasets:
            shutil.move(f"/root/{lora_name}/{i.dataset_name}",i.dataset_dir_path) 
        shutil.rmtree(f"/root/{lora_name}") 
         
        print(f"{allow_patterns} successfully downloaded from {repo_id}") 
        
    except Exception:
        print(f"The dataset is not found. Run push_dataset() first:")
        print(f"python .py push_dataset")