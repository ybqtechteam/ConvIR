import cv2
from typing import List
import os
from numpy import ndarray
import torch.nn.functional as f
from torchvision.transforms import functional as F
from .model import *
import tqdm
import numpy as np


def video2frames(video_path: str, target_fps: float) -> List[ndarray]:
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Impossibile aprire il video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)  
    print(f"Original video FPS: {video_fps}")

    frame_interval = int(video_fps / target_fps) 
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        frame_count += 1
    
    cap.release()
    return frames











if __name__ == "__main__":
    VIDEO_PATH = "demo/10.mp4"
    FPS = 1

    model_path = MODEL_TYPE.DEHAZE

    if model_path == MODEL_TYPE.DERAIN:
        model = DerainConvIR(model_path.value)
    
    elif model_path == MODEL_TYPE.DEHAZE or model_path == MODEL_TYPE.DEHAZE_BASE:
        model = DehazeConvIR(model_path.value, version='small')
    
    elif model_path == MODEL_TYPE.DESNOW:
        model = DesnowConvIR(model_path.value)

    # os.makedirs("demo/tmp", exist_ok=True)
    # os.makedirs("demo/image/clean", exist_ok=True)

    frames = video2frames(VIDEO_PATH, target_fps=FPS)

    # for i, frame in enumerate(tqdm.tqdm(frames, desc="Saving ground truth frames")):
    #     cv2.imwrite(f"demo/image/gt/{i:03d}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    print(f"Total {len(frames)} frames extracted")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    

    for i, frame in enumerate(tqdm.tqdm(frames, desc="Processing frames")):

        if i == 0:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(VIDEO_PATH.replace(".mp4", "_CLEANED.mp4"), fourcc, FPS, (640, 480))

        out = model(frame)
        writer.write(cv2.cvtColor(np.array(out), cv2.COLOR_BGR2RGB))
        out.save(f"demo/tmp/{i:03d}.png")
    
    writer.release()
