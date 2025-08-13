# scripts/extract_fuse_features.py

import os
import cv2
import torch
import librosa
import numpy as np
from torchvision import models, transforms
from glob import glob
from pathlib import Path
import pandas as pd
from moviepy.editor import VideoFileClip
from scripts.config import PROJECT_ROOT, DATA_DIR, OUTPUTS_DIR, MODELS_DIR, FEATURE_DIR

# Set paths
AUDIO_DIR = OUTPUTS_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_LIMIT = 2  # ProofOfConcept: Number of files to process 
SAMPLE_LIMIT = None  # None to process all files 



# Load MobileNetV2 pretrained model (no classification head)
# Device placement
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet.classifier = torch.nn.Identity()
mobilenet.to(device)
mobilenet.eval()

# Transform for video frames
transform = transforms.Compose([
    transforms.ToPILImage(),
    
    # Resize to match MobileNetV2 input size: 224x224 pixels
    transforms.Resize((224, 224)),  

    # convert to tensor of shape (3, 224, 224) : 3 is number of channels (RGB colors)
    transforms.ToTensor(),

    # normalize each channel: they match the data distribution expected by models pretrained on ImageNet
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

def extract_video_features(video_path):

    # Open video for reading``
    cap = cv2.VideoCapture(video_path)

    # video frames list
    frames = []

    # read all frames and append them to the list
    while True:
        # ret is False if there are no more frames
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()

    if len(frames) == 0:
        return None
    
    # get 10 evenly spaced indices. Sample up to 10 evenly spaced frames
    sampled_frames = np.linspace(0, len(frames) - 1, num=10, dtype=int)
    
    # list of features from each of sampled video frames
    features = []

    # preprocess each sampled frame and extract features
    for idx in sampled_frames:
        frame = frames[idx]
        
        # apply transform pipeline transofrations
        frame_tensor = transform(frame).unsqueeze(0).to(device)   # Move to device. 1 batch with shape: (1, 3, 224, 224)
        
        # Disable gradient tracking and extract features using MobileNetV2
        with torch.no_grad():
            feat = mobilenet(frame_tensor).squeeze(0).numpy()  # Move back to CPU before numpy. shape: (1280,)  1D vecor with 1280 elements
        

        features.append(feat)

    return np.stack(features)  # shape: (10, 1280)

def extract_audio_from_video(video_path, audio_path):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)


def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    # Extract MFCC features,  computing 13 MFCC coefficients per frame
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.T  # shape: (num_frames, 13)

def pad_features(feat, target_len):
    if len(feat) >= target_len:
        return feat[:target_len]
    else:
        pad_width = ((0, target_len - len(feat)), (0, 0))
        # Pad row dimension (number of features) with zeros to the end of the feature array 
        return np.pad(feat, pad_width, mode='constant')

def fuse_features(video_feat, audio_feat):
    target_len = max(len(video_feat), len(audio_feat))
    video_feat = pad_features(video_feat, target_len)
    audio_feat = pad_features(audio_feat, target_len)
    return np.concatenate((video_feat, audio_feat), axis=1)  # shape: (target_len, combined_dim)


def main():

    print(f"Output directory Features -> {FEATURE_DIR}")
    print(f"Output directory Audio cache -> {AUDIO_DIR}")

    ravdess_label_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
    }

    ravdess_label_records = []



    # Recursively find all mp4 files under Actor_* folders
    mp4_files = glob(os.path.join(DATA_DIR, "**", "*.mp4"), recursive=True)
   
    if SAMPLE_LIMIT:
        mp4_files = mp4_files[:SAMPLE_LIMIT]
    print(f"Found {len(mp4_files)} mp4 files.")
    
    

    for path in mp4_files:
        filename = os.path.basename(path)
        
        # Skip 'video only' or 'audio only' files
        if filename.split("-")[0] != "01":
            print(f"\nSkipped non multimodal file: {filename}")
            continue  
        
        print(f"\nProcessing: {filename}")


        # Extract label from filename
        emotion_code = filename.split("-")[2]
        emotion_label = ravdess_label_map.get(emotion_code, "unknown")

       
        # Extract features
        print("     Extracting features...")
        try:

            # 1) Extract audio features
            video_path = path
            audio_path = AUDIO_DIR / (filename.replace(".mp4", ".wav"))  # Save audio as .wav
            
            # Check if audio file already exists and extract (create) if not
            if not audio_path.exists():                
                extract_audio_from_video(video_path, audio_path)   
            
            # Extract audio features from audo .wav file                 
            audio_feat = extract_audio_features(audio_path)
            
            if audio_feat is None:
                print("  Skipping: no audio features found.")
                continue
                   
            # Ensure audio features are at least 10 frames
            if audio_feat.shape[0] < 10:
                print("  Skipping: not enough audio frames.")
                continue

            print(f"    Audio MFCC shape: {audio_feat.shape}")


            # 2) Extract video features
            video_feat = extract_video_features(path)
            if video_feat is None:
                print("  Skipping: no video frames found.")
                continue

            # Ensure video features are at least 10 frames
            if video_feat.shape[0] < 10:
                print("  Skipping: not enough video frames.")
                continue

            print(f"    Video frame features shape: {video_feat.shape}")

            # 3) Fuse features
            fused_feat = fuse_features(video_feat, audio_feat)
            print(f"    Fused feature shape: {fused_feat.shape}")

            # 4) Save fused features to a file
            output_file = FEATURE_DIR / (filename.replace(".mp4", ".npy"))      # PROJECT_ROOT/outputs/features/xxxxx.npy
            np.save(output_file, fused_feat)

            # Save label info
            ravdess_label_records.append({"file": filename.replace(".mp4", ".npy"), "label": emotion_label})


        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            continue
    



  

    #5) All files are processed. Save labels to CSV
    df = pd.DataFrame(ravdess_label_records)
    df.to_csv(PROJECT_ROOT / "outputs" / "ravdess_labels.csv", index=False)
    print("\n✅ Completed feature extraction and label saving.")


if __name__ == "__main__":
    main()
