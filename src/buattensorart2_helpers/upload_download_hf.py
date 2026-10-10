from huggingface_hub import snapshot_download,hf_hub_download
import os
from tqdm.auto import tqdm
import subprocess
import shutil
from pathlib import Path

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
        



def snapshot_comfy(models_dir:str,repo_id:str,allow_patterns:dict,ignore_patterns:list=[],max_workers=8):
    
    allow_patterns_list=[hf_path for hf_path,symlink_path in allow_patterns.items()]
    
    local_dir=Path(f"{models_dir}/{repo_id}")
    local_dir.mkdir(parents=True,exist_ok=True)
    
    
    try:
        models=snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            ignore_patterns=ignore_patterns,
            allow_patterns=allow_patterns_list,
            max_workers=max_workers,
            tqdm_class=tqdm
        )
        print(f"downloaded to {local_dir}")
        
    except Exception as e:
        print(f"failed to download {repo_id}")
        print(f"error : {e}")
    
    for hf_path,symlink_path in allow_patterns.items():
        try:
            subprocess.run(
                f"ln -s {models}/{hf_path} {symlink_path}",
                shell=True,
                check=True
            )
            
        except Exception as e:
            print(f"failed to make symlink for {hf_path}")
            print(f"error : {e}")
    
            
def hf_hub_comfy(models_dir:str,repo_id:str,filename:dict):
    
    local_dir=Path(f"{models_dir}/{repo_id}")
    local_dir.mkdir(parents=True,exist_ok=True)
    
    for hf_path,symlink_path in filename.items():
        
        try:
            models=hf_hub_download(
                repo_id=repo_id,
                local_dir=local_dir,
                filename=hf_path,
                tqdm_class=tqdm
            )
            print(f"downloaded to {local_dir}")
            
        except Exception as e:
            print(f"failed to download {repo_id}")
            print(f"error : {e}")
    
    
        try:
            subprocess.run(
                f"ln -s {models} {symlink_path}",
                shell=True,
                check=True
            )
            
        except Exception as e:
            print(f"failed to make symlink for {hf_path}")
            print(f"error : {e}")