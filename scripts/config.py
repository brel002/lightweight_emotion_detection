# scripts/config.py
from pathlib import Path
import os

# Project root = repo root (…/emotion_detection)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Storage root = where to store data, outputs, models, etc.
STORAGE_ROOT = Path(os.getenv("ED_STORAGE", PROJECT_ROOT/"storage"))

# From Troch
#BACKBONE_NAME = "mobilenet_v2"                                         # for torch implementation: manually change the extract file to use it


# use timm to get models for extraction
#BACKBONE_NAME = "mobilenetv2_100"                                     
#BACKBONE_NAME = "mobilenetv3_small_100"                                
#BACKBONE_NAME = "mobilenetv4_conv_small_050.e3000_r224_in1k"                           
#BACKBONE_NAME = "mobilevit_s.cvnets_in1k"                             
#BACKBONE_NAME = "efficientnet_b0"                                      
#BACKBONE_NAME = "mobilenetv3_large_100"                                
#BACKBONE_NAME = "deit_tiny_patch16_224"                                #

BACKBONE_NAME = "bruno_test"
# Dimensions of the features: extracted features with prosody concatenated to audio features
VIDEO_DIM = 1280    
AUDIO_DIM = 1280 

# from torch
if BACKBONE_NAME == "mobilenet_v2" :                                     
    VIDEO_DIM = 1280    
    AUDIO_DIM = 1280    


# from timm
if BACKBONE_NAME == "mobilenetv2_100" :                                   
    VIDEO_DIM = 1280     
    AUDIO_DIM = 1280     
      
if BACKBONE_NAME == "efficientnet_b0":                                   
    AUDIO_DIM = 1280 
    VIDEO_DIM = 1280 

if BACKBONE_NAME == "mobilenetv3_small_100":                                      
    VIDEO_DIM = 1024    
    AUDIO_DIM = 1024

if BACKBONE_NAME == "mobilenetv3_large_100":        
    VIDEO_DIM = 1280    
    AUDIO_DIM = 1280

if BACKBONE_NAME == "mobilenetv4_conv_small_050.e3000_r224_in1k":                                  
    VIDEO_DIM = 1280     
    AUDIO_DIM = 1280

if BACKBONE_NAME == "mobilevit_s.cvnets_in1k":                                   
    VIDEO_DIM = 640    
    AUDIO_DIM = 640

if BACKBONE_NAME == "deit_tiny_patch16_224":                                  
    VIDEO_DIM = 192    
    AUDIO_DIM = 192

if BACKBONE_NAME == "bruno_test":                                  
    VIDEO_DIM = 1    
    AUDIO_DIM = 20



DATA_DIR     = STORAGE_ROOT / "data"                            # RAVDESS mp4s
MODELS_DIR   = STORAGE_ROOT / "models" / BACKBONE_NAME          # best trained models, overwritten with each run
OUTPUTS_DIR  = STORAGE_ROOT / "outputs"                         # outputs of the scripts




# use this when training/testing/ablation
FEATURE_DIR   = OUTPUTS_DIR / "features" / BACKBONE_NAME 



# normalization stats from packed feature fies
PACKED_FUSED_CONCAT_FEATURE_NORM_POOLED_NPZ = FEATURE_DIR / "packed_fused_concat_feature_norm_pooled.npz"  
PACKED_FUSED_INTERLEAVED_FEATURE_NORM_POOLED_NPZ = FEATURE_DIR / "packed_fused_intlv_feature_norm_pooled.npz"  
PACKED_AUDIO_FEATURE_NORM_POOLED_NPZ = FEATURE_DIR / "packed_audio_feature_norm_pooled.npz" 
PACKED_VIDEO_FEATURE_NORM_POOLED_NPZ = FEATURE_DIR / "packed_video_feature_norm_pooled.npz" 

PACKED_FUSED_CONCAT_FEATURE_NORM_FRAMES_NPZ = FEATURE_DIR / "packed_fused_concat_feature_norm_frames.npz"
PACKED_FUSED_INTERLEAVED_FEATURE_NORM_FRAMES_NPZ = FEATURE_DIR / "packed_fused_intlv_feature_norm_frames.npz"
PACKED_AUDIO_FEATURE_NORM_FRAMES_NPZ = FEATURE_DIR / "packed_audio_feature_norm_frames.npz"
PACKED_VIDEO_FEATURE_NORM_FRAMES_NPZ = FEATURE_DIR / "packed_video_feature_norm_frames.npz"



# default train/val/test splits (actor-wise)
TRAIN_ACTORS = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
VAL_ACTORS = [5,6, 7, 8]
TEST_ACTORS  = [1, 2, 3, 4]





# Create dirs if not exist
MODELS_DIR.mkdir(parents=True, exist_ok=True) 
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_DIR.mkdir(parents=True, exist_ok=True)





  


