import cv2
import argparse
import babyros
import base64
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Camera Image Publisher for BabyROS")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--topic", type=str, default="camera_image", help="Topic to publish images to")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    
    # Camera settings
    parser.add_argument("--brightness", type=float, default=-1.0, help="Camera brightness (0-255, -1 to ignore)")
    parser.add_argument("--contrast", type=float, default=-1.0, help="Camera contrast (0-255, -1 to ignore)")
    parser.add_argument("--sharpness", type=float, default=-1.0, help="Camera sharpness (0-255, -1 to ignore)")
    parser.add_argument("--gain", type=float, default=-1.0, help="Camera gain/ISO (0-255, -1 to ignore)")
    parser.add_argument("--exposure", type=float, default=-1.0, help="Camera exposure (-1 to ignore)")
    parser.add_argument("--camera-settings", action="store_true", help="Open native camera properties dialog")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    pub_image = babyros.node.Publisher(topic=args.topic)
    print(f"Publishing to topic: '{args.topic}'")
    
    cap = cv2.VideoCapture(args.camera, cv2.CAP_ANY)
    if not cap.isOpened():
        for idx in [0, 2, 3]:
            cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
            if cap.isOpened():
                break
    if not cap.isOpened():
        print("Error: Could not open any camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if args.camera_settings:
        print("Opening native camera settings dialog...")
        cap.set(cv2.CAP_PROP_SETTINGS, 1)
        
    if args.brightness >= 0:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, args.brightness)
    if args.contrast >= 0:
        cap.set(cv2.CAP_PROP_CONTRAST, args.contrast)
    if args.sharpness >= 0:
        cap.set(cv2.CAP_PROP_SHARPNESS, args.sharpness)
    if args.gain >= 0:
        cap.set(cv2.CAP_PROP_GAIN, args.gain)
    if args.exposure >= 0:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
        cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened at {frame_width}x{frame_height}")
    
    frame_period = 1.0 / args.fps

    try:
        while True:
            start_time = time.perf_counter()

            success, image = cap.read()
            if not success:
                print("Failed to grab frame. Retrying...")
                time.sleep(0.1)
                continue

            # Encode and publish the image
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            success_encode, encoded_image = cv2.imencode(".jpg", image, encode_param)

            if success_encode:
                img_bytes = encoded_image.tobytes()
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")

                img_msg = {
                    "header": {
                        "stamp": {
                            "sec": int(time.time()),
                            "nanosec": int((time.time() % 1) * 1e9),
                        },
                        "frame_id": "camera",
                    },
                    "format": "jpeg",
                    "data": img_b64,
                }

                pub_image.publish(img_msg)

            elapsed = time.perf_counter() - start_time
            sleep_time = frame_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nPublisher interrupted by user.")
    finally:
        cap.release()
        print("Publisher cleanup complete.")

if __name__ == "__main__":
    main()