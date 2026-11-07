
def make_payload_dict(
    prompt:str,
    use_convrot:bool=True,
    lora_bypass_strength:float=1.3,
    width:int=1280,
    height:int=1280,
    use_flash_attn:bool=True,
    lora_name:str="alamakjang",
    seed:int=42,
    steps:int=10,
    bypass_or_not:bool=True
    ) -> dict:

  
    mydict={
    "1": {
        "inputs": {
        "unet_name": "krea2_turbo_int8_convrot.safetensors" if use_convrot == True else "krea2_turbo_bf16.safetensors",
        "weight_dtype": "default"
        },
        "class_type": "UNETLoader",
        "_meta": {
        "title": "Load Diffusion Model"
        }
    },
    "2": {
        "inputs": {
        "clip_name": "qwen3vl_4b_int8_convrot.safetensors" if use_convrot == True else "qwen3vl_4b_bf16.safetensors",
        "type": "krea2",
        "device": "default"
        },
        "class_type": "CLIPLoader",
        "_meta": {
        "title": "Load CLIP"
        }
    },
    "3": {
        "inputs": {
        "vae_name": "wan_2.1_vae.safetensors"
        },
        "class_type": "VAELoader",
        "_meta": {
        "title": "Load VAE"
        }
    },
    "4": {
        "inputs": {
        "width": width,
        "height": height,
        "batch_size": 1
        },
        "class_type": "EmptyLatentImage",
        "_meta": {
        "title": "Empty Latent Image"
        }
    },
    "5": {
        "inputs": {
        "lora_01": "snofs_krea_v1_nostrip.safetensors" if bypass_or_not == True else "None",
        "strength_01": lora_bypass_strength if bypass_or_not == True else 1,
        "lora_02": lora_name,
        "strength_02": 1,
        "lora_03": "None",
        "strength_03": 1,
        "lora_04": "None",
        "strength_04": 1,
        "model": [
            "1",
            0
        ],
        "clip": [
            "2",
            0
        ]
        },
        "class_type": "Lora Loader Stack (rgthree)",
        "_meta": {
        "title": "Lora Loader Stack (rgthree)"
        }
    },
    "6": {
        "inputs": {
        "allow_compile": True,
        "model": [
            "5",
            0
        ]
        },
        "class_type": "PatchFlashAttentionKJ",
        "_meta": {
        "title": "Patch Flash Attention KJ"
        }
    },
    "7": {
        "inputs": {
        "text": prompt,
        "clip": [
            "5",
            1
        ]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {
        "title": "CLIP Text Encode (Prompt)"
        }
    },
    "8": {
        "inputs": {
        "conditioning": [
            "7",
            0
        ]
        },
        "class_type": "ConditioningZeroOut",
        "_meta": {
        "title": "Conditioning Zero Out"
        }
    },
    "9": {
        "inputs": {
        "seed": seed,
        "steps": steps,
        "cfg": 1,
        "sampler_name": "euler",
        "scheduler": "beta",
        "denoise": 1,
        "model": [
            "15",
            0
        ],
        "positive": [
            "7",
            0
        ],
        "negative": [
            "8",
            0
        ],
        "latent_image": [
            "4",
            0
        ]
        },
        "class_type": "KSampler",
        "_meta": {
        "title": "KSampler"
        }
    },
    "11": {
        "inputs": {
        "samples": [
            "9",
            0
        ],
        "vae": [
            "3",
            0
        ]
        },
        "class_type": "VAEDecode",
        "_meta": {
        "title": "VAE Decode"
        }
    },
    "15": {
        "inputs": {
        "condition": use_flash_attn,
        "value_if_true": [
            "6",
            0
        ],
        "value_if_false": [
            "5",
            0
        ]
        },
        "class_type": "_ConditionalSelect",
        "_meta": {
        "title": "Conditional Select"
        }
    },
    "16": {
        "inputs": {
        "images": [
            "11",
            0
        ]
        },
        "class_type": "SaveImageWebsocket",
        "_meta": {
        "title": "Save Image (Websocket)"
        }
    }
    }
    
    return mydict