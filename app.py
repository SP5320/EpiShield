import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from flask import Flask, request, redirect, url_for, render_template,session, send_from_directory, flash
import pandas as pd
UPLOAD_FOLDER = 'uploaded_videos'
PROCESSED_FOLDER = 'processed_videos'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)
    
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.secret_key = 'your_secret_key_here'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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




def predict_flashes(video_frames_luminance, fps, file_without_extension, thres = 0.2, dark_threshold_percentage=0.6, flashing_pixels_percentage=0.5, lookahead_frames=10, time_buffer=3):
    """Predict and mark potential flashes in the video using dynamic thresholds without region-based analysis."""
    flash_timestamps = []
    current_flash_range_start = None
    last_end_time = 0
    flash_frames = []
    frame_indices = []
    luminance_changes = []
    
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
            flash_frames.append(i)

        if flash_detected:
            if current_flash_range_start is None:
                current_flash_range_start = i / fps
            last_end_time = (i + lookahead_frames) / fps
        else:
            if current_flash_range_start is not None and (i / fps - last_end_time) > time_buffer:
                flash_timestamps.append((current_flash_range_start, last_end_time))
                current_flash_range_start = None
                
        frame_indices.append(i)
        luminance_changes.append(np.sum(luminance_diff) / luminance_diff.size)
        
        
    if current_flash_range_start is not None:
        flash_timestamps.append((current_flash_range_start, len(video_frames_luminance) / fps))
    
    plt.figure(figsize=(10, 5))
    plt.plot(frame_indices, luminance_changes, label='Luminance Change')
    plt.scatter(flash_frames, [luminance_changes[i] for i in flash_frames], color='red', label='Detected Flashes')  # Marking the flash frames
    plt.axhline(y=luminance_change_threshold * 255, color='green', linestyle='--', label='Threshold')
    plt.xlabel('Frame Index')
    plt.ylabel('Average Luminance Change')
    plt.title('Luminance Change Over Frames')
    plt.legend()
    output_plot = f'processed_videos/{file_without_extension}/graph'
    if not os.path.exists(output_plot):
        os.makedirs(output_plot)
    plt.savefig(os.path.join(output_plot, f'luminance_graph_{file_without_extension}.jpeg'))
    return flash_timestamps



def save_video(video_path, filename, sensitive_ranges, mode, file_without_extension):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    output_dir = f'processed_videos/{file_without_extension}/{mode}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    out = cv2.VideoWriter(os.path.join(f'processed_videos/{file_without_extension}/{mode}', f'processed_{filename}'), fourcc, fps, (width, height))

    def process_frame(frame, mode, darken_factor=0.3):
        if mode == 'grayscale':
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dark_frame = cv2.convertScaleAbs(gray_frame, alpha=darken_factor, beta=0)
            return cv2.cvtColor(dark_frame, cv2.COLOR_GRAY2BGR)
        else:
            return frame
    in_sensitive_range = False
    current_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reached end of video or can't receive frame (stream end?). Exiting ...")
            break
        
        # Calculate the current time from the frame number
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # convert to seconds
        for start, end in sensitive_ranges:
            if start - 1 <= current_time < end:
                in_sensitive_range = True
                if mode == 'skip':
                    # Skip to end of the sensitive range
                    cap.set(cv2.CAP_PROP_POS_MSEC, end * 1000)  # Move to the end of the current sensitive range
                    continue
                elif mode == 'grayscale':
                    frame = process_frame(frame, 'grayscale')
                    continue

        if not in_sensitive_range:
            frame = process_frame(frame, 'normal')
            
        out.write(frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video processing completed")

    
@app.route('/', methods = ['GET','POST'])
# ‘/’ URL is bound with hello_world() function.
def index():
    return render_template('upload.html',filename="",download=False)


@app.route('/upload', methods = ['GET','POST'])
def upload():
    if 'file' not in request.files: 
        return 'No video file found'
    video = request.files['file']
    mode = request.form['mode']
    if video.filename == "":
        return 'No video selected'
    if video and allowed_file(video.filename):
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
        video.save(video_path)
        video_path = "./"+video_path
        fps = get_video_fps(video_path)
        video_luminance = preprocess_video(video_path)
        filename = 'processed_'+video.filename
        file_without_extension = filename.split('.')[0]
        sensitive_flash_ranges = predict_flashes(video_luminance, fps, file_without_extension)
        sensitive = pd.DataFrame(sensitive_flash_ranges, columns =['Start Time(Seconds)', 'End Time(Seconds)'])
        table = sensitive.to_html(classes='table-style', index=True)
        print("Sensitive flashes: ",sensitive_flash_ranges)
        save_video(video_path, video.filename, sensitive_flash_ranges, mode, file_without_extension)
        session['dir'] = f'processed_videos/{file_without_extension}/{mode}'
        flash('Video processed successfully. It is now available for download.')
        return render_template('upload.html', filename=filename, download=True, table = table)
    return "Invalid Format"

@app.route('/download-video/<filename>')
def download_video(filename):
    directory = session['dir'] # Specify the directory where video is stored
    filename = filename           # Specify the filename of the video
    return send_from_directory(directory, filename, as_attachment=True)

if __name__ == '__main__':
	app.run(host='localhost',port='8000',threaded = True)



