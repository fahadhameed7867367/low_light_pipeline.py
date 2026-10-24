import cv2
import numpy as np
import os

def adjust_gamma(image, gamma=1.0):
    """Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values."""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def enhance_image(image):
    """Applies a classical restoration pipeline (Gamma Correction + CLAHE) to recover details."""
    # 1. Gamma correction to brighten shadows
    brightened = adjust_gamma(image, gamma=2.0)
    
    # 2. Convert to LAB color space to apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L-channel (lightness)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge back and convert to BGR
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_img

def detect_faces(img, title):
    """Runs Haar Cascade detection and returns the image with boxes drawn."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    print(f"[{title}] Found {len(faces)} face(s).")
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    return img, len(faces)

if __name__ == "__main__":
    # 1. Load a normal image
    # REPLACE 'sample.jpg' with the path to any clear face image you have
    image_path = "sample.jpg" 
    original_img = cv2.imread(image_path)
    
    if original_img is None:
        print(f"Error: Could not load {image_path}. Please provide a valid image.")
        exit()

    # 2. Simulate Low Light (Darken the image significantly)
    # We divide by 3 to make it very dark
    low_light_img = (original_img / 3.0).astype(np.uint8)
    
    # 3. Run detection on the Low-Light Image (Expect Failure)
    failed_img, failed_count = detect_faces(low_light_img.copy(), "Low-Light (Before Restoration)")
    
    # 4. Apply Restoration (Enhancement)
    restored_img = enhance_image(low_light_img)
    
    # 5. Run detection on the Restored Image (Expect Success)
    success_img, success_count = detect_faces(restored_img.copy(), "Restored (After Enhancement)")
    
    # 6. Save the results
    if not os.path.exists("results"):
        os.makedirs("results")
        
    cv2.imwrite("results/1_low_light_failure.jpg", failed_img)
    cv2.imwrite("results/2_restored_success.jpg", success_img)
    
    print("\nPipeline complete. Check the 'results' folder.")
    if success_count > failed_count:
        print("Conclusion: Restoration successfully recovered the faces lost in the dark.")
    else:
        print("Note: If counts are the same, try a darker image or tweak the gamma value.")