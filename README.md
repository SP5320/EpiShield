# EpiShield: Detecting Seizure Inducing Scenes In Videos

## Description
This app processes uploaded video files to detect and mitigate seizure-inducing effects. Users can upload videos, which the app will then preprocess to identify potential seizure-triggering content. The user can choose to download the video in grayscale or skip sensitive content based on the identified frames.

## Features
- Upload video files (.mp4, .mov, .avi).
- Process videos to detect potential seizure-inducing frames.
- Convert videos to grayscale or allow skipping of sensitive sections.
- Download processed videos.

### Prerequisites
- Python 3.8 or later
- Flask
- OpenCV
- NumPy
- Matplotlib

### Install dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### Setting Up
**Clone the repository**
```bash
git clone https://github.iu.edu/cs-b657-sp2024/cs-b657-sp2024-dasakas-kparekh-ks134-pawars-finalproject.git
```

### Configure the App
- Open `app.py` and set the `app.secret_key` to a secret key of your choice.

## Running the Application

### Start the server
Navigate to the project directory and run:
```bash
python app.py
```

### Access the Web Interface
- Open a web browser and visit http://localhost:8000.
- Follow the web interface to upload and process videos.

## Usage
- **Upload Video**: Click the 'Upload' button and select a video file.
- **Select Processing Mode**: Choose whether to convert the video to grayscale or skip sensitive sections.
- **Download Processed Video**: After processing, a download link will be provided. Click to download the modified video.

