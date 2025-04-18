from utils.tracker import Tracker
from utils.video_utils import read_video, save_video

def main():
    # read video
    video_frames = read_video('videos/input_videos/match.mp4')
    
    # initialize tracker
    tracker = Tracker(model_path="models/best.pt")
    
    tracks = tracker.get_object_tracks(frames=video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    
    # draw annotations
    output_video_frames = tracker.draw_annotations(video_frames=video_frames, tracks=tracks)
    
    # save video
    save_video(output_video_frames=output_video_frames, output_video_path='videos/output_videos/output_video.avi')

if __name__ == "__main__":
    main()
