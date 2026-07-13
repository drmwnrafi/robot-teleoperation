import cv2
import numpy as np
import glob
import os
import time
import platform

CAMERA_INDEX = 0                  
FRAME_WIDTH = 1920              
FRAME_HEIGHT = 1080
N_IMAGES = 25                     
IMAGE_DIR = "calibration_results"

MIRROR_DISPLAY = True             # True for front-facing webcams
SHARPNESS_THRESHOLD = 40          
AUTO_CAPTURE_COOLDOWN = 1.5      

CHESSBOARD_SIZE = (10, 7)         
SQUARE_SIZE = 22.0                
OUTPUT_FILE = f"cam_calib_{CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]}_{SQUARE_SIZE}mm_{FRAME_WIDTH}x{FRAME_HEIGHT}.npz"

for f in glob.glob(os.path.join(IMAGE_DIR, "*.jpg")):
    os.remove(f)
os.makedirs(IMAGE_DIR, exist_ok=True)

def capture_images():
    print(f"\n=== HANDS-FREE CAPTURING ({N_IMAGES} IMAGES) ===")
    print("Instructions:")
    print("1. Move the board to different areas of the screen.")
    print("2. Hold it still. The camera will AUTO-CAPTURE when in focus!")
    print("3. Watch the 'Focus Score' on screen. It must exceed the threshold.")
    print("4. Press Q to quit.\n")
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        if platform.system() == "Linux":
            cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
            
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        if platform.system() == "Linux":
            print("NOTE: If you are using WSL2, USB cameras are not passed through by default.")
            print("You must attach the camera to WSL using 'usbipd'.")
        return False
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    
    captured = 0
    last_capture_time = 0
    show_flash_until = 0
    coverage_grid = np.zeros((3, 3), dtype=bool)
    
    while captured < N_IMAGES:
        ret, frame = cap.read()
        if not ret:
            continue
            
        h, w = frame.shape[:2]
        current_time = time.time()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_sharp = laplacian > SHARPNESS_THRESHOLD
        ret_corners = False
        corners = None
        
        if is_sharp:
            small_gray = cv2.resize(gray, (w // 2, h // 2))
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
            ret_corners, small_corners = cv2.findChessboardCorners(small_gray, CHESSBOARD_SIZE, flags)
            
            if ret_corners:
                corners = small_corners * 2.0
        
        display_frame = cv2.flip(frame, 1) if MIRROR_DISPLAY else frame.copy()
        
        for i in range(1, 3):
            cv2.line(display_frame, (w*i//3, 0), (w*i//3, h), (100, 100, 100), 1)
            cv2.line(display_frame, (0, h*i//3), (w, h*i//3), (100, 100, 100), 1)
            
        for i in range(3):
            for j in range(3):
                if coverage_grid[i, j]:
                    x1, y1 = w*j//3, h*i//3
                    x2, y2 = w*(j+1)//3, h*(i+1)//3
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                    cv2.addWeighted(overlay, 0.2, display_frame, 0.8, 0, display_frame)

        is_ready = is_sharp and ret_corners
        border_color = (0, 255, 0) if is_ready else (0, 0, 255)
        cv2.rectangle(display_frame, (0, 0), (w-1, h-1), border_color, 5)

        if ret_corners:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            
            if MIRROR_DISPLAY:
                m_corners = corners.copy()
                m_corners[:, 0, 0] = w - m_corners[:, 0, 0]
                cv2.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, m_corners, ret_corners)
                board_center = np.mean(m_corners, axis=0)[0]
            else:
                cv2.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, corners, ret_corners)
                board_center = np.mean(corners, axis=0)[0]

            grid_x = min(2, int(board_center[0] / (w / 3)))
            grid_y = min(2, int(board_center[1] / (h / 3)))
            coverage_grid[grid_y, grid_x] = True

        cv2.putText(display_frame, f"Captured: {captured}/{N_IMAGES}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        focus_text = "FOCUSED - HOLD STILL" if is_sharp else "BLURRY - ADJUST FOCUS"
        focus_color = (0, 255, 0) if is_sharp else (0, 0, 255)
        cv2.putText(display_frame, focus_text, (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, focus_color, 2)
                    
        score_color = (0, 255, 0) if is_sharp else (0, 165, 255)
        cv2.putText(display_frame, f"Focus Score: {laplacian:.0f} / {SHARPNESS_THRESHOLD}", (20, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_color, 2)

        if not ret_corners and is_sharp:
            cv2.putText(display_frame, "Chessboard not detected!", (20, 180), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        if current_time < show_flash_until:
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
            cv2.putText(display_frame, "CAPTURED!", (w//2 - 150, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

        cv2.imshow('Auto-Capture Calibration', display_frame)
        
        if is_ready and (current_time - last_capture_time > AUTO_CAPTURE_COOLDOWN):
            filename = f"{IMAGE_DIR}/calib_{captured:03d}.jpg"
            cv2.imwrite(filename, frame) # Save ORIGINAL, unmirrored full-res frame
            print(f"Auto-captured {filename} (Focus Score: {laplacian:.0f})")
            captured += 1
            last_capture_time = current_time
            show_flash_until = current_time + 0.5
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Cancelled by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    
    if captured < 10:
        print(f"\nWARNING: Only captured {captured} images. 15-25 is recommended.")
        return False
        
    print(f"\nSuccessfully captured {captured} images.\n")
    return True

def calculate_calibration():
    print(f"=== CALCULATING CALIBRATION ===\n")
    
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    
    objpoints = []
    imgpoints = []
    images = sorted(glob.glob(f"{IMAGE_DIR}/*.jpg"))
    
    gray = None
    for fname in images:
        img = cv2.imread(fname)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        try:
            ret, corners = cv2.findChessboardCornersSB(gray, CHESSBOARD_SIZE)
        except AttributeError:
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, flags)
            
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print(f"  ⚠ Failed to find corners in {os.path.basename(fname)}")
    
    if len(objpoints) == 0:
        print("ERROR: No chessboard corners found in any image!")
        return False
        
    print(f"✓ Found corners in {len(objpoints)}/{len(images)} images.")
    print("Calculating camera matrix...\n")
    
    image_size = gray.shape[::-1]
    flags = cv2.CALIB_FIX_K3 
    
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None, flags=flags
    )
    
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)

    print("="*50)
    print("CALIBRATION RESULTS")
    print("="*50)
    print(f"RMS Error : {ret:.4f} px")
    print(f"Mean Error: {mean_error:.4f} px (Should be < 0.5 for good results)")
    print(f"\nCamera Matrix:\n{camera_matrix}")
    print(f"\nDistortion Coefficients:\n{dist_coeffs.ravel()}")
    
    np.savez(OUTPUT_FILE, 
             camera_matrix=camera_matrix, 
             dist_coeffs=dist_coeffs,
             image_width=image_size[0],
             image_height=image_size[1])
    print(f"\nCalibration saved to: {OUTPUT_FILE}")
    print("="*50)
    
    print("\n=== VISUAL VERIFICATION ===")
    print("Showing side-by-side comparison of Distorted vs Undistorted.")
    print("Press any key to cycle through images, or 'q' to finish.\n")
    
    h, w = image_size[::-1]
    newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
    
    target_display_height = 400 
    
    for fname in images[:5]: 
        img = cv2.imread(fname)
        if img is None: continue
        
        if MIRROR_DISPLAY:
            img = cv2.flip(img, 1)
        
        dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramatrix)
        
        x, y, w_roi, h_roi = roi
        if w_roi > 0 and h_roi > 0:
            dst_cropped = dst[y:y+h_roi, x:x+w_roi]
        else:
            dst_cropped = dst
            
        scale_orig = target_display_height / img.shape[0]
        disp_orig = cv2.resize(img, None, fx=scale_orig, fy=scale_orig)
        
        scale_dist = target_display_height / dst_cropped.shape[0]
        disp_dist = cv2.resize(dst_cropped, None, fx=scale_dist, fy=scale_dist)
        
        combined = np.hstack((disp_orig, disp_dist))
        
        cv2.putText(combined, "Original", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(combined, "Undistorted", (int(disp_orig.shape[1]) + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Verification (Press any key)', combined)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()
    return True

def main():
    print("\n" + "="*50)
    print("ADVANCED AUTO-CAPTURE CALIBRATION TOOL")
    print("="*50)
    
    if not capture_images():
        return
        
    if not calculate_calibration():
        return
        
    print("\nCalibration complete! You can now use the .npz file in your tracking scripts.")

if __name__ == "__main__":
    main()