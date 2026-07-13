import cv2
import numpy as np
import base64
import babyros
import time

def image_callback(msg):
    """Callback function triggered every time a new image message is received."""
    
    # 1. Extract the Base64 string from the message dictionary
    b64_data = msg.get("data")
    if not b64_data:
        return

    try:
        # 2. Decode the Base64 string back into raw JPEG bytes
        jpg_bytes = base64.b64decode(b64_data)
        
        # 3. Convert the raw bytes into a NumPy array
        nparr = np.frombuffer(jpg_bytes, np.uint8)
        
        # 4. Decode the NumPy array into an OpenCV BGR image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 5. Display the image
        if img is not None:
            cv2.imshow("Compressed Image Subscriber", img)
            cv2.waitKey(1) # Required to keep the OpenCV window responsive
            
    except Exception as e:
        print(f"Error decoding image: {e}")

def main():
    print("✓ Starting 'image_compressed' subscriber...")
    
    # Initialize the subscriber
    # Note: Adjust the initialization if your babyros API requires a specific message type
    sub = babyros.node.Subscriber(topic="image_compressed", callback=image_callback)
    
    print("✓ Waiting for images... Press 'q' in the image window to quit.")
    
    # Keep the script running and processing callbacks
    try:
        while True:
            # If your babyros library has a blocking spin function (like babyros.node.spin()), 
            # you can replace this while loop with that function.
            time.sleep(0.01) 
    except KeyboardInterrupt:
        print("\nShutting down subscriber...")
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()