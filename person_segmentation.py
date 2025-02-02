import cv2
import mediapipe as mp
import numpy as np
import os

# Add MediaPipe pose initialization
mp_pose = mp.solutions.pose

def create_pose_mask(image_shape, landmarks, padding=30):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    h, w = image_shape[:2]
    
    # Convert landmarks to points
    points = []
    for landmark in landmarks.landmark:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        points.append([x, y])
    points = np.array(points, dtype=np.int32)
    
    # Create initial hull
    hull = cv2.convexHull(points)
    
    # Create separate mask for legs to ensure better coverage
    leg_landmarks = [
        landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP],
        landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP],
        landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE],
        landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE],
        landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE],
        landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE],
        landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX],
        landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
    ]
    
    leg_points = []
    for landmark in leg_landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        # Add points with horizontal padding for better leg coverage
        leg_points.append([x - padding, y])
        leg_points.append([x + padding, y])
    leg_points = np.array(leg_points, dtype=np.int32)
    
    # Draw filled polygons
    cv2.fillConvexPoly(mask, hull, 255)
    if len(leg_points) > 0:
        leg_hull = cv2.convexHull(leg_points)
        cv2.fillConvexPoly(mask, leg_hull, 255)
    
    # Dilate the mask to ensure coverage
    kernel = np.ones((padding, padding), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Apply Gaussian blur to smooth the edges
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    
    return mask

def process_video_segmentation(video_path):
    # Initialize MediaPipe
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=2)
    segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    # Read the video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writers
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    outputs_dir = "outputs"
    
    # Create outputs directory if it doesn't exist
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    
    # Define output paths in the outputs directory
    output_overlay_path = os.path.join(outputs_dir, f'{base_name}_segmentation_overlay.mp4')
    output_mask_path = os.path.join(outputs_dir, f'{base_name}_mask_only.mp4')
    output_person_path = os.path.join(outputs_dir, f'{base_name}_person_extracted.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_overlay = cv2.VideoWriter(output_overlay_path, fourcc, fps, (frame_width, frame_height))
    out_mask = cv2.VideoWriter(output_mask_path, fourcc, fps, (frame_width, frame_height))
    out_person = cv2.VideoWriter(output_person_path, fourcc, fps, (frame_width, frame_height))

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

        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get both pose and segmentation
        pose_results = pose.process(image_rgb)
        seg_results = segmentation.process(image_rgb)
        
        if seg_results.segmentation_mask is not None:
            # Get the basic segmentation mask
            seg_mask = seg_results.segmentation_mask
            seg_mask = (seg_mask > 0.1).astype(np.uint8) * 255
            
            # If pose is detected, use it to refine the mask
            if pose_results.pose_landmarks:
                pose_mask = create_pose_mask(image.shape, pose_results.pose_landmarks)
                
                # Combine masks: take maximum of both masks
                final_mask = cv2.max(seg_mask, pose_mask)
            else:
                final_mask = seg_mask
            
            # Convert mask to 3 channels
            mask_3ch = np.stack((final_mask,) * 3, axis=-1) / 255.0
            
            # Create outputs (working in RGB space)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            overlay_color = np.zeros_like(image_rgb)
            overlay_color[:] = (0, 255, 0)  # Green in RGB
            
            # Create overlay
            overlay_image = (cv2.addWeighted(image_rgb, 1, overlay_color, 0.5, 0) * 
                           mask_3ch).astype(np.uint8)
            
            # Create person extraction
            person_extracted = (image_rgb * mask_3ch).astype(np.uint8)
            
            # Convert back to BGR for saving
            overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)
            person_extracted = cv2.cvtColor(person_extracted, cv2.COLOR_RGB2BGR)
            
            # Write frames
            out_overlay.write(overlay_image)
            out_mask.write((mask_3ch * 255).astype(np.uint8))
            out_person.write(person_extracted)

    # Release everything
    cap.release()
    out_overlay.release()
    out_mask.release()
    out_person.release()
    segmentation.close()
    
    print(f"\nProcessing complete. Outputs saved to:")
    print(f"Overlay: {output_overlay_path}")
    print(f"Mask: {output_mask_path}")
    print(f"Extracted Person: {output_person_path}")

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
            process_video_segmentation(video_path)