import os, argparse

import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_video_path', type=str)
    parser.add_argument('--output-path', type=str, default="video_frames")
    parser.add_argument('--frame-interval', type=int, default=5)

    args = parser.parse_args()
    return args


def main(args):
    input_video_path = args.input_video_path
    output_frames_dir = args.output_path
    frame_interval = int(args.frame_interval) # Extract every Nth frame

    if not os.path.exists(output_frames_dir):
        os.makedirs(output_frames_dir)

    print(f"Opening video: {input_video_path}")
    vidcap = cv2.VideoCapture(input_video_path)
    if not vidcap.isOpened():
        raise RuntimeError(f"Error: Could not open video file {input_video_path}")

    count = 0
    frame_idx = 0
    success, image = vidcap.read()

    while success:
        if frame_idx % frame_interval == 0:
            frame_filename = os.path.join(output_frames_dir, f"frame_{count:06d}.jpg")
            cv2.imwrite(frame_filename, image)     # save frame as JPEG file
            print(f"Extracted frame {count:06d} (video frame {frame_idx})")
            count += 1
        success, image = vidcap.read()
        frame_idx += 1

    vidcap.release()
    print(f"Finished extracting {count} frames to {output_frames_dir}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
