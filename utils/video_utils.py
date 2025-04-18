import cv2

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