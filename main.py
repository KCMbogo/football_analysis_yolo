from utils.tracker import Tracker
from utils.video_utils import read_video, save_video, run_realtime_video

def main():
    # Choose mode: 'save' or 'realtime'
    mode = 'realtime'  
    
    if mode == 'save':
        # read video
        video_frames = read_video('videos/input_videos/match.mp4')
        
        # initialize tracker
        tracker = Tracker(model_path="models/nano.pt")
        
        tracks = tracker.get_object_tracks(frames=video_frames,
                                         read_from_stub=True,
                                         stub_path='stubs/track_stubs.pkl')
        
        # draw annotations
        output_video_frames = tracker.draw_annotations(video_frames=video_frames, tracks=tracks)
        
        # save video
        save_video(output_video_frames=output_video_frames, output_video_path='videos/output_videos/output_video.avi')
    else:
        # Run real-time video processing
        run_realtime_video(video_path='videos/input_videos/match.mp4', model_path="models/large.pt")

if __name__ == "__main__":
    main()