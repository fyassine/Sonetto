#!/usr/bin/env python3
"""
Face Registration Script

This script allows you to register a face for a new user in the system.
It will capture an image from your webcam or use a specified image file.
"""

import sys
import argparse
from memory_module.user_identification import register_face

def main():
    parser = argparse.ArgumentParser(description='Register a face for a user.')
    parser.add_argument('user_id', type=str, help='User ID for the face to register')
    parser.add_argument('--image', type=str, help='Path to image file (optional)', default=None)
    parser.add_argument('--no-webcam', action='store_true', help='Do not use webcam')
    
    args = parser.parse_args()
    
    if args.no_webcam and not args.image:
        print("Error: If --no-webcam is set, you must provide an image file path.")
        return 1
    
    print(f"Registering face for user: {args.user_id}")
    
    use_webcam = not args.no_webcam
    
    if args.image:
        # Read the image file
        try:
            with open(args.image, 'rb') as f:
                image_data = f.read()
            
            import base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            success = register_face(args.user_id, face_image_base64=image_base64)
        except Exception as e:
            print(f"Error processing image file: {str(e)}")
            success = False
    else:
        # Use webcam
        success = register_face(args.user_id, use_camera=True)
    
    if success:
        print(f"Successfully registered face for user: {args.user_id}")
        return 0
    else:
        print(f"Failed to register face for user: {args.user_id}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
