import cv2
import mediapipe as mp
import os
mp_pose = mp.solutions.pose

def process_video(video_path):
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    # Read the video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    outputs_dir = "outputs"
    
    # Create outputs directory if it doesn't exist
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    
    output_path = os.path.join(outputs_dir, f'{base_name}_with_pose.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
            
        frame_count += 1
        progress = (frame_count / total_frames) * 100
        
        # Create loading bar
        bar_width = 50
        filled_width = int(bar_width * frame_count / total_frames)
        bar = '=' * filled_width + '-' * (bar_width - filled_width)
        print(f'\rProcessing: [{bar}] {progress:.1f}%', end='')

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect poses
        results = pose.process(image_rgb)

        # Draw pose landmarks on the image
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

        # Write the frame to output video
        out.write(image)

    # Release everything
    cap.release()
    out.release()
    pose.close()
    print(f"\nProcessing complete. Output saved to: {output_path}")

if __name__ == "__main__":
    # Create necessary directories if they don't exist
    videos_dir = "videos"
    outputs_dir = "outputs"
    
    if not os.path.exists(videos_dir):
        os.makedirs(videos_dir)
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
        
    # Process all videos in the videos folder
    for video_file in os.listdir(videos_dir):
        if video_file.lower().endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(videos_dir, video_file)
            print(f"Processing: {video_path}")
            process_video(video_path)