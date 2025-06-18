import cv2
import numpy as np
from tkinter import *
from tkinter import messagebox

# import slack
import os
from pathlib import Path
# from dotenv import load_dotenv

# env_path = Path('.')/ '.env'
# load_dotenv(dotenv_path=env_path)

# client = slack.WebClient(token=os.environ['SLACK_TOKEN'])
# client.chat_postMessage(channel='#epilepsy_detector', text="Hello from python script!")


def get_video_fps(video_path):
    """Get the frames per second of the video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def preprocess_video(video_path):
    """Preprocess the video to calculate and store the luminance of each frame."""
    cap = cv2.VideoCapture(video_path)
    video_frames_luminance = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_luminance = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        video_frames_luminance.append(frame_luminance)

    cap.release()
    return video_frames_luminance

def dynamic_threshold(luminance_frame):
    """Calculate dynamic threshold based on the frame's average luminance."""
    avg_luminance = np.mean(luminance_frame)
    if avg_luminance < 50:  # Darker frame
        return 0.03  # Lower threshold for dark scenes
    elif avg_luminance > 200:  # Brighter frame
        return 0.07  # Higher threshold for bright scenes
    else:
        return 0.05  # Standard threshold

def predict_flashes(video_frames_luminance, fps, dark_threshold_percentage=0.6, flashing_pixels_percentage=0.5, lookahead_frames=10, time_buffer=3):
    """Predict and mark potential flashes in the video using dynamic thresholds without region-based analysis."""
    flash_timestamps = []
    current_flash_range_start = None
    last_end_time = 0

    for i in range(len(video_frames_luminance) - lookahead_frames):
        flash_detected = False
        current_frame = video_frames_luminance[i]
        future_frame = video_frames_luminance[i + lookahead_frames]
        
        luminance_change_threshold = dynamic_threshold(current_frame)
        
        luminance_diff = cv2.absdiff(future_frame, current_frame)
        max_luminance = max(future_frame.max(), current_frame.max())
        threshold_diff = luminance_change_threshold * max_luminance
        dark_threshold_value = dark_threshold_percentage * 255
        dark_threshold = np.where(future_frame < dark_threshold_value, 1, 0)
        change_threshold = np.where(luminance_diff > threshold_diff, 1, 0)
        flashing_pixels = np.bitwise_and(dark_threshold.astype(np.uint8), change_threshold.astype(np.uint8))
        
        if np.sum(flashing_pixels) / flashing_pixels.size > flashing_pixels_percentage:
            flash_detected = True

        if flash_detected:
            if current_flash_range_start is None:
                current_flash_range_start = i / fps
            last_end_time = (i + lookahead_frames) / fps
        else:
            if current_flash_range_start is not None and (i / fps - last_end_time) > time_buffer:
                flash_timestamps.append((current_flash_range_start, last_end_time))
                current_flash_range_start = None

    if current_flash_range_start is not None:
        flash_timestamps.append((current_flash_range_start, len(video_frames_luminance) / fps))

    return flash_timestamps


def play_video(video_path, sensitive_ranges):
    window = Tk()
    window.title("Video Player")
    Label(window, text="Video Player").pack()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)

    def process_frame(frame, mode, darken_factor=0.3):
        if mode == 'grayscale':
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dark_frame = cv2.convertScaleAbs(gray_frame, alpha=darken_factor, beta=0)
            return cv2.cvtColor(dark_frame, cv2.COLOR_GRAY2BGR)
        else:
            return frame

    def on_close():
        cap.release()
        window.destroy()

    def custom_dialog():
        dialog = Toplevel(window)
        dialog.title("Video Playback")
        Label(dialog, text="Potential Sensitive content detected. Choose an option to move forward with:").pack(pady=10)
        user_choice = None

        def set_choice(choice):
            nonlocal user_choice
            user_choice = choice
            dialog.destroy()

        Button(dialog, text="Skip", command=lambda: set_choice('skip')).pack(side="left", padx=10, pady=10)
        Button(dialog, text="Grayscale", command=lambda: set_choice('grayscale')).pack(side="left", padx=10, pady=10)
        Button(dialog, text="Cancel", command=lambda: set_choice(None)).pack(side="left", padx=10, pady=10)

        dialog.grab_set()  
        window.wait_window(dialog)  
        return user_choice

    user_choice = custom_dialog()
    if user_choice is None:
        on_close()
        return
    mode = user_choice

    def play():
        ret, frame = cap.read()
        if not ret:
            on_close()
            return
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        in_sensitive_range = False
    
        # Check if the current time is within any sensitive ranges
        for start, end in sensitive_ranges:
            if start - 1 <= current_time < end:
                in_sensitive_range = True
                if mode == 'skip':
                    cap.set(cv2.CAP_PROP_POS_MSEC, end * 1000)  # Skip to end of sensitive range
                    break
                elif mode == 'grayscale':
                    frame = process_frame(frame, 'grayscale')
                    break
    
        if not in_sensitive_range:
            frame = process_frame(frame, 'normal')

        cv2.imshow('Frame', frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            on_close()
            return
        window.after(delay, play)  # Schedule the next frame based on the video delay


    window.protocol("WM_DELETE_WINDOW", on_close)
    play()
    window.mainloop()

video_path = './uploaded_videos/WhatsApp Video 2024-04-24 at 15.18.10.mp4'
fps = get_video_fps(video_path)
video_luminance = preprocess_video(video_path)
sensitive_flash_ranges = predict_flashes(video_luminance, fps)
print("Sensitive flashes: ",sensitive_flash_ranges)

play_video(video_path, sensitive_flash_ranges)

