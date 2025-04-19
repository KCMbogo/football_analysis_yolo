import cv2
import supervision as sv
from .tracker import Tracker

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        success, img = cap.read()
        if not success:
            break
        frames.append(img)
    return frames

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    h, w = output_video_frames[0].shape[0], output_video_frames[0].shape[1]
    out = cv2.VideoWriter(filename=output_video_path, fourcc=fourcc, fps=24, frameSize=(w, h))
    for frame in output_video_frames:
        out.write(frame)
    out.release()

def run_realtime_video(video_path, model_path):
    # Initialize the video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return
    
    # Get video properties
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # frame_delay = int(1000 / fps)  # Delay between frames in milliseconds
    
    # Initialize tracker
    tracker = Tracker(model_path=model_path)
    
    # Process video frame by frame
    frame_buffer = []
    frame_count = 0
    batch_size = 5  
    
    print("Press 'q' to quit...")
    
    while True:
        # Read a frame
        success, frame = cap.read()
        if not success:
            break
            
        frame = cv2.resize(src=frame, dsize=(1280, 720))
        frame_buffer.append(frame)
        frame_count += 1
        
        # When we have enough frames in the buffer, process them
        if len(frame_buffer) >= batch_size:
            # Use the existing get_object_tracks function to handle detection and tracking
            tracks = tracker.get_object_tracks(frames=frame_buffer, read_from_stub=False)
            
            # Draw annotations on frames and display them
            annotated_frames = tracker.draw_annotations(video_frames=frame_buffer, tracks=tracks)
            
            # Display each annotated frame
            for i, annotated_frame in enumerate(annotated_frames):
                # Display frame number
                cv2.putText(annotated_frame, f"Frame: {frame_count - len(frame_buffer) + i + 1}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display the frame
                cv2.imshow("Real-time Object Tracking", annotated_frame)
                
                # Wait for the appropriate amount of time to maintain video speed
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
            
            # Clear the buffer
            frame_buffer = []
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Video processing complete")
