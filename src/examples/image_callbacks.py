import cv2
import numpy as np
import base64
import babyros
import time

def image_callback(msg):
    b64_data = msg.get("data")
    if not b64_data:
        return
    try:
        jpg_bytes = base64.b64decode(b64_data)
        nparr = np.frombuffer(jpg_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            cv2.imshow("Compressed Image Subscriber", img)
            cv2.waitKey(1)
    except Exception as e:
        print(f"Error decoding image: {e}")

def main():
    print("Starting 'image_compressed' subscriber...")
    sub = babyros.node.Subscriber(topic="image_compressed", callback=image_callback)
    print("✓ Waiting for images... Press 'q' in the image window to quit.")
    try:
        while True:
            time.sleep(0.01) 
    except KeyboardInterrupt:
        print("\nShutting down subscriber...")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()