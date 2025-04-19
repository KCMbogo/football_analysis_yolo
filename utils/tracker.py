import supervision as sv
import pickle, os, cv2
from ultralytics import YOLO
from utils.bbox_utils import get_bbox_width, get_center_of_bbox

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path).to(device="cuda")
        self.tracker = sv.ByteTrack()
        
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames=frames)
        
        tracks = {
            "players": [],
            "referees": [],
            "ball": [],
        }
        
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k, v in cls_names.items()}
            
            # convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # convert goalkeeper to player object
            for object_id, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_id] = cls_names_inv['player']
                    
            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            tracks['players'].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                
                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                    
                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                        
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}
                    
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
                    
        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox=bbox)
        width = get_bbox_width(bbox=bbox)
        
        cv2.ellipse(img=frame, 
                    center=(x_center, y2), 
                    axes=(int(width), int(0.53*width)), 
                    angle=0.0, 
                    startAngle=-45, 
                    endAngle=235, 
                    color=color, thickness=2, lineType=cv2.LINE_4)
        
        rect_width = 40
        rect_height = 20
        x1_rect = x_center - rect_width//2
        x2_rect = x_center - rect_width//2
        y1_rect = (y2 - rect_height//2) + 15
        y2_rect = (y2 + rect_height//2) + 15
        
        if track_id is not None:
            cv2.rectangle(img=frame, 
                          pt1=(int(x1_rect), int(y1_rect)),
                          pt2=(int(x2_rect), int(y2_rect)),
                          color=color,
                          thickness=cv2.FILLED)
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            cv2.putText(img=frame,
                        text=f"{str(track_id)}", 
                        org=(int(x1_rect), int(y1_rect + 15)), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=0.6, 
                        color=(0, 0, 0), 
                        thickness=2)
                
        
        return frame        
    
    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            
            # annotate player
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player['bbox'], (0, 0, 255), track_id)
                
            # annotate referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0, 255, 255))
            
            output_video_frames.append(frame)
        return output_video_frames