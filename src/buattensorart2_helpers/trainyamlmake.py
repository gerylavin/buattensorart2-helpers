import yaml
import os

def trainYamlMake(config_data:dict,arch:str,config_file_path:str):
    '''
    to create train.yaml, a config file for finetuning using ai-tookit on google colab, kaggle. etc.
    
    '''
    
    model = config_data["config"]["process"][0]["model"]
    
    match arch:
        case "zimage:turbo":
            REPO_MODEL="Tongyi-MAI/Z-Image-Turbo"
            model.update({"assistant_lora_path":"ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors"})
            model['name_or_path']=REPO_MODEL
        case "zimage":
            REPO_MODEL="Tongyi-MAI/Z-Image"
            model['name_or_path']=REPO_MODEL
        case "flux2_klein_9b":
            REPO_MODEL="black-forest-labs/FLUX.2-klein-base-9B"
            model['name_or_path']=REPO_MODEL
        case "flux2_klein_4b":
            REPO_MODEL="black-forest-labs/FLUX.2-klein-base-4B"
            model['name_or_path']=REPO_MODEL
        case "ernie_image":
            REPO_MODEL="baidu/ERNIE-Image"
            model['name_or_path']=REPO_MODEL
        case "qwen_image_edit_plus:2511":
            REPO_MODEL="Qwen/Qwen-Image-Edit-2511"
            model['name_or_path']=REPO_MODEL
        case "qwen_image:2512":
            REPO_MODEL="Qwen/Qwen-Image-2512"
            model['name_or_path']=REPO_MODEL
        case "ideogram4":
            REPO_MODEL="ideogram-ai/ideogram-4-fp8"
            model['name_or_path']=REPO_MODEL
        case "boogu_image":
            REPO_MODEL="Boogu/Boogu-Image-0.1-Base"
            model['name_or_path']=REPO_MODEL
        case "boogu_image_edit":
            REPO_MODEL="Boogu/Boogu-Image-0.1-Edit"
            model['name_or_path']=REPO_MODEL
        case "krea2":
            REPO_MODEL="krea/Krea-2-Raw"
            model['name_or_path']=REPO_MODEL
 
        case "zimage_l2p":
            REPO_MODEL="zhen-nan/L2P"
            filename="model-1k-merge.safetensors"
            model.update({"extras_name_or_path": "Tongyi-MAI/Z-Image-Turbo"})
            model['name_or_path']=REPO_MODEL
        case "qwen_image_2":
            REPO_MODEL="Comfy-Org/Qwen-Image-2.1"
            model['name_or_path']=REPO_MODEL
        case "ming_image":
            model.update({"assistant_lora_path":"ostris/ming_image_training_adapter/ming_image_01_design_training_adapter_v1.safetensors"})
            model['name_or_path']="Kijai/Ming-Image-ComfyUI"
        case "minimax_h3":
            REPO_MODEL="Comfy-Org/MiniMax-H3"
            model['name_or_path']=REPO_MODEL
            
            model.update({"assistant_lora_path":"ostris/minimax_h3_training_adapter/minimax_h3_training_adapter_v1.safetensors"})
            config_data["config"]["process"][0]["network"]['network_kwargs']['ignore_if_contains']=["adaln_proj"]
        case "qwen_image_2:turbo":
            REPO_MODEL="Comfy-Org/Qwen-Image-2.1/diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors"
            model.update({"assistant_lora_path":"ostris/qwen_image_2_turbo_training_adapter/qwen_image_2_turbo_training_adapter_v1.safetensors"})
            
            model["name_or_path"]=REPO_MODEL
        case "iris3b":
            REPO_MODEL="speridlabs/iris-3b"
            model["name_or_path"]=REPO_MODEL
        case "flux3":
            REPO_MODEL="black-forest-labs/flux-3-image-klein-base"
            model["name_or_path"]=REPO_MODEL
        case "moss_image":
            REPO_MODEL="OpenMOSS-Team/MOSS-Image-SFT"
            model["name_or_path"]=REPO_MODEL
            
            model["network_kwargs"]["ignore_if_contains"]=["layers.22."]
            model["model_kwargs"]={"uncond_skip_layers":[22]}
        case "fibo2":
            REPO_MODEL="Comfy-Org/Fibo-2.0"
            model['name_or_path']=REPO_MODEL
            model["model_kwargs"]={"match_target_res":True}
        case _:
            raise Exception("pilih arsitektur (ARCH) yg bener")

    try:
        output_dir = os.path.dirname(config_file_path)
        if output_dir:  
            os.makedirs(output_dir, exist_ok=True)
     
        with open(config_file_path, 'w') as f:
 
            yaml.dump(config_data, f, sort_keys=False, default_flow_style=False)
            
        print(f"file konfigurasi di: {config_file_path}\n")
        
    except Exception as e:
        print(f"Error: {e}")