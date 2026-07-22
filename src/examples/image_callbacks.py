import cv2
import numpy as np
import base64
import babyros
import time

# Global variable to hold the latest decoded frame
latest_frame = None

def image_callback(msg):
    global latest_frame
    b64_data = msg.get("data")
    if not b64_data:
        return
    try:
        jpg_bytes = base64.b64decode(b64_data)
        nparr = np.frombuffer(jpg_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is not None:
            # JUST save the frame here. DO NOT call cv2.imshow in the callback!
            latest_frame = img
    except Exception as e:
        print(f"Error decoding image: {e}")

def main():
    global latest_frame
    print("Starting 'image_compressed' subscriber...")
    
    # 1. Create and show the blank image initially
    blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imshow("Compressed Image Subscriber", blank_img)
    cv2.waitKey(1)

    sub = babyros.node.Subscriber(topic="image_compressed", callback=image_callback)
    print("✓ Waiting for images... Press 'q' in the image window to quit.")
    
    try:
        while True:
            # 2. If a new frame arrived, display it in the MAIN thread
            if latest_frame is not None:
                cv2.imshow("Compressed Image Subscriber", latest_frame)
                # Optional: set to None after displaying so it only updates on new frames
                # latest_frame = None 
            
            # 3. Process GUI events and check for 'q' key in the MAIN thread
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            time.sleep(0.01) 
    except KeyboardInterrupt:
        print("\nShutting down subscriber...")
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()