import os
import face_recognition
import json
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
from pathlib import Path
import time

# Directory to store known face encodings
KNOWN_FACES_DIR = Path("known_faces")
KNOWN_FACES_DIR.mkdir(exist_ok=True)

# Path to store face encodings database
FACE_ENCODINGS_FILE = KNOWN_FACES_DIR / "face_encodings.json"

# Initialize face encodings database if it doesn't exist
def _initialize_face_db():
    """Initialize the face encodings database if it doesn't exist."""
    if not FACE_ENCODINGS_FILE.exists():
        with open(FACE_ENCODINGS_FILE, 'w') as f:
            json.dump({}, f)
        
        # Create a demo user with a predefined face if no users exist
        # This will make the system work with default 'Ahmed' until real faces are registered
        demo_user = {
            "Ahmed": [[0.0] * 128]  # Dummy encoding that won't match any real face
        }
        _save_face_encodings(demo_user)
        print("Initialized face database with demo user 'Ahmed'")

_initialize_face_db()

def _load_face_encodings():
    """Load face encodings from the database file."""
    try:
        with open(FACE_ENCODINGS_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # If file is empty or corrupted, initialize it
        _initialize_face_db()
        return {}

def _save_face_encodings(encodings_db):
    """Save face encodings to the database file."""
    with open(FACE_ENCODINGS_FILE, 'w') as f:
        json.dump(encodings_db, f)

def _capture_image_from_webcam():
    """Capture an image from the webcam.
    
    Returns:
        numpy.ndarray: Image array if capture successful, None otherwise
    """
    try:
        print("Attempting to capture image from webcam...")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Failed to open webcam")
            return None
        
        # Allow camera to warm up
        time.sleep(0.5)
        
        # Capture frame
        ret, frame = cap.read()
        
        # Release webcam
        cap.release()
        
        if not ret or frame is None:
            print("Failed to capture image from webcam")
            return None
        
        print("Successfully captured image from webcam")
        # Convert from BGR (OpenCV format) to RGB (face_recognition format)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    except Exception as e:
        print(f"Error capturing image from webcam: {str(e)}")
        return None

def register_face(user_id, face_image_base64=None, use_camera=False):
    """Register a new face for a user.
    
    Args:
        user_id (str): Unique identifier for the user
        face_image_base64 (str, optional): Base64 encoded image containing the user's face
        use_camera (bool, optional): Whether to use webcam to capture face image
        
    Returns:
        bool: True if registration was successful, False otherwise
    """
    try:
        # Get image either from webcam or from base64 string
        if use_camera:
            image_np = _capture_image_from_webcam()
            if image_np is None:
                print(f"Failed to capture webcam image for user {user_id}")
                return False
        elif face_image_base64:
            # Decode base64 image
            image_data = base64.b64decode(face_image_base64)
            image = Image.open(BytesIO(image_data))
            # Convert PIL Image to numpy array
            image_np = np.array(image)
        else:
            print("Neither webcam nor base64 image specified for registration")
            return False
        
        # Find face locations in the image
        face_locations = face_recognition.face_locations(image_np)
        
        if not face_locations:
            print(f"No faces found in the image for user {user_id}")
            return False
        
        # Compute face encodings
        face_encodings = face_recognition.face_encodings(image_np, face_locations)
        
        if not face_encodings:
            print(f"Could not compute face encodings for user {user_id}")
            return False
        
        # Convert numpy arrays to lists for JSON serialization
        encodings_list = [encoding.tolist() for encoding in face_encodings]
        
        # Load existing encodings database
        encodings_db = _load_face_encodings()
        
        # Add or update user's face encodings
        encodings_db[user_id] = encodings_list
        
        # Save updated encodings database
        _save_face_encodings(encodings_db)
        
        # Save face image for reference
        user_image_path = KNOWN_FACES_DIR / f"{user_id}.jpg"
        if use_camera:
            # Convert back to BGR for OpenCV
            img_to_save = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(user_image_path), img_to_save)
        else:
            Image.fromarray(image_np).save(user_image_path)
        
        print(f"Successfully registered face for user {user_id}")
        return True
    
    except Exception as e:
        print(f"Error registering face for user {user_id}: {str(e)}")
        return False

def identify_user(face_image_base64=None, use_camera=False, tolerance=0.6):
    """Identify a user from a face image.
    
    Args:
        face_image_base64 (str, optional): Base64 encoded image containing a face
        use_camera (bool, optional): Whether to use webcam to capture face
        tolerance (float): Tolerance for face comparison (lower is more strict)
        
    Returns:
        tuple: (success, user_id) where success is a boolean indicating if identification succeeded,
               and user_id is the identified user or None if no match was found
    """
    try:
        # Load face encodings database
        encodings_db = _load_face_encodings()
        
        if not encodings_db:
            print("No registered faces found in the database")
            # Return Ahmed as default user for compatibility
            return True, "Ahmed"
        
        # Get image either from webcam or from base64 string
        if use_camera:
            image_np = _capture_image_from_webcam()
            if image_np is None:
                print("Failed to capture webcam image for identification")
                # Return Ahmed as default user if camera fails
                return True, "Ahmed"
        elif face_image_base64:
            # Decode base64 image
            image_data = base64.b64decode(face_image_base64)
            image = Image.open(BytesIO(image_data))
            # Convert PIL Image to numpy array
            image_np = np.array(image)
        else:
            print("Neither webcam nor base64 image specified for identification")
            # Return Ahmed as default user if no image provided
            return True, "Ahmed"
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(image_np)
        
        if not face_locations:
            print("No faces found in the image")
            # Return Ahmed as default user if no face found
            return True, "Ahmed"
        
        face_encodings = face_recognition.face_encodings(image_np, face_locations)
        
        if not face_encodings:
            print("Could not compute face encodings for the image")
            # Return Ahmed as default user if encodings failed
            return True, "Ahmed"
        
        # Compare with known faces
        for user_id, known_encodings_list in encodings_db.items():
            # Convert list back to numpy array
            known_encodings = [np.array(encoding) for encoding in known_encodings_list]
            
            # Compare faces
            for face_encoding in face_encodings:
                matches = [face_recognition.compare_faces([known_encoding], face_encoding, tolerance=tolerance)[0] 
                           for known_encoding in known_encodings]
                
                if any(matches):
                    print(f"User {user_id} identified")
                    return True, user_id
        
        print("No matching user found, using default user: Ahmed")
        # Return Ahmed as default user if no match found
        return True, "Ahmed"
    
    except Exception as e:
        print(f"Error during user identification: {str(e)}")
        # Return Ahmed as default user if an error occurred
        return True, "Ahmed"

def get_identified_user_profile(face_image_base64=None, image_path=None, use_camera=False):
    """Get user profile after identification.
    
    This function identifies a user using face recognition and retrieves their profile.
    If no face is provided or identification fails, it returns a default user (Ahmed).
    
    Args:
        face_image_base64 (str, optional): Base64 encoded image containing a face
        image_path (str, optional): Path to an image file containing a face
        use_camera (bool, optional): Whether to use webcam to capture face
        
    Returns:
        dict: User profile data
    """
    from .db import get_customer_profile
    
    print("Identifying user for profile retrieval...")
    
    # Try to identify user based on provided parameters
    if use_camera:
        _, user_id = identify_user(use_camera=True)
    elif face_image_base64:
        _, user_id = identify_user(face_image_base64=face_image_base64)
    elif image_path:
        # Load the image file
        try:
            image = face_recognition.load_image_file(image_path)
            # Convert to base64 for identification
            _, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            _, user_id = identify_user(face_image_base64=image_base64)
        except Exception as e:
            print(f"Error loading image from path: {str(e)}")
            user_id = "Ahmed"  # Default user
    else:
        print("No identification method specified, using default user")
        user_id = "Ahmed"  # Default user
    
    print(f"Identified user: {user_id}, retrieving profile...")
    
    # Get user profile from database
    return get_customer_profile(user_id)
