from pydantic import BaseModel,computed_field, Field, model_validator

class myDataset(BaseModel):
    dataset_local_path:str = Field(exclude=True)     
    cache_latents_to_disk:bool=True
    resolution:list=[512]
    num_frames:int=1
    do_i2v:bool=False
    do_audio:bool=False
    auto_frame_count:bool=False
    caption_dropout_rate:float=0.05
    default_caption:str=""
    is_reg:bool=False
    fps:int=24
    controls:list=[]
    shrink_video_to_frames:bool=True
    flip_x:bool=False
    flip_y:bool=False
    num_repeats:int=1
    control_path_local_1:str=None
    control_path_local_2:str=None
    control_path_local_3:str=None
    
    machine:int=1
    
    @model_validator(mode='after')
    def whatmachine(self):
        match self.machine:
            case 2:
                self.dataset_local_path=f"./{self.dataset_local_path.rstrip('/').split('/')[-1]}"
        return self
    
    @property
    def dataset_name(self) -> str:
        return self.dataset_local_path.rstrip('/').split('/')[-1]

    @property
    def dataset_dir_path(self) -> str:
        return f"/root/{self.dataset_name}"
    
    @computed_field
    @property
    def folder_path(self) -> str:
        return self.dataset_dir_path
    
    @computed_field
    @property
    def control_path_1(self) -> str:
        return self.control_path_local_1
    
    @computed_field
    @property
    def control_path_2(self) -> str:
        return self.control_path_local_2
    
    @computed_field
    @property
    def control_path_3(self) -> str:
        return self.control_path_local_3