from __future__ import print_function

import time
import cv2
import numpy as np
import random
import dlib
import sys

# Import all functions from eye_key_funcs
from eye_key_funcs import (
    init_camera, 
    adjust_frame, 
    make_black_page, 
    make_white_page, 
    show_window, 
    shut_off, 
    display_box_around_face, 
    display_face_points, 
    get_eye_coordinates, 
    display_eye_lines, 
    is_blinking, 
    find_cut_limits, 
    pupil_on_cut_valid, 
    project_on_page, 
    dysplay_keyboard, 
    identify_key, 
    take_radius_eye
)

from projected_keyboard import get_keyboard

# ------------------------------------ Inputs
camera_ID = 0  # select webcam

width_keyboard, height_keyboard = 2000,  1000 # [pixels]
offset_keyboard = (50, 50) # pixel offset (x, y) of keyboard coordinates

resize_eye_frame = 5.5 # scaling factor for window's size
resize_frame = 0.4 # scaling factor for window's size
# ------------------------------------
def calculate_keyboard_dimensions(size_screen):
    # Use a significant portion of the screen width and height
    width_keyboard = int(size_screen[1] * 0.9)  # 90% of screen width
    height_keyboard = int(size_screen[0] * 0.6)  # 60% of screen height
    
    # Calculate offset to center the keyboard
    offset_x = int((size_screen[1] - width_keyboard) / 2)
    offset_y = int((size_screen[0] - height_keyboard) / 2)
    
    return width_keyboard, height_keyboard, (offset_x, offset_y)
def safe_detect_faces(gray_frame, detector, confidence_threshold=0.5):
    """
    Safely detect faces with confidence threshold
    
    Args:
        gray_frame (numpy.ndarray): Grayscale input frame
        detector (dlib.simple_object_detector): Face detector
        confidence_threshold (float): Minimum confidence for face detection
    
    Returns:
        list: List of detected faces
    """
    # Detect faces
    faces = detector(gray_frame)
    
    # If multiple faces detected, choose the largest
    if len(faces) > 1:
        print(f'Multiple faces detected: {len(faces)}. Using largest face.')
        # Sort faces by area and select the largest
        faces = sorted(faces, key=lambda face: (face.right() - face.left()) * (face.bottom() - face.top()), reverse=True)
    
    return faces[:1]  # Return only the first/largest face

def ensure_full_frame_capture(camera):
    """
    Ensure the camera is capturing full frames
    
    Args:
        camera (cv2.VideoCapture): Camera object
    
    Returns:
        bool: True if frame capture is successful, False otherwise
    """
    # Try different resolution settings
    resolutions = [
        (1920, 1080),  # Full HD
        (1280, 720),   # HD
        (640, 480)     # Standard
    ]
    
    for width, height in resolutions:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        ret, frame = camera.read()
        if ret and frame is not None and frame.size > 0:
            print(f"Successfully capturing frames at {width}x{height}")
            return True
    
    print("Unable to capture full frames")
    return False

def main():
    # Initialize the camera
    camera = init_camera(camera_ID)
    
    if not ensure_full_frame_capture(camera):
        print("Camera setup failed")
        return

    # take size screen
    size_screen = (int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)), 
                   int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)))

    width_keyboard, height_keyboard, offset_keyboard = calculate_keyboard_dimensions(size_screen)

    # make a page (2D frame) to write & project
    keyboard_page = make_black_page(size = size_screen)
    calibration_page = make_black_page(size = size_screen)

    # Initialize keyboard
    key_points = get_keyboard(width_keyboard  = width_keyboard,
                               height_keyboard = height_keyboard,
                               offset_keyboard = offset_keyboard)

    # upload face/eyes predictors
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Calibration corners
    corners = [
    (50, 50),  # Top Left Corner with 50px offset
    (size_screen[1]-50, size_screen[0]-50),  # Bottom Right Corner with 50px offset
    (size_screen[1]-50, 50),  # Top Right Corner with 50px offset
    (50, size_screen[0]-50)   # Bottom Left Corner with 50px offset
    ]
    calibration_cut = []
    corner = 0

    while(corner < 4):  # calibration of 4 corners
        ret, frame = camera.read()   # Capture frame
        if not ret or frame is None:
            print("Frame capture failed")
            break

        frame = adjust_frame(frame)  # rotate / flip

        gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # gray-scale to work with

        # messages for calibration
        cv2.putText(calibration_page, 'Calibration: Look at Green Circle and Blink', 
                    (50, 100),  # Fixed position near top 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.circle(calibration_page, corners[corner], 40, (0, 255, 0), -1)


        # detect faces in frame with safe detection
        faces = safe_detect_faces(gray_scale_frame, detector)

        if faces:
            face = faces[0]  # Get the first/largest face
            display_box_around_face(frame, [face.left(), face.top(), face.right(), face.bottom()], 'green', (20, 40))

            landmarks = predictor(gray_scale_frame, face) # find points in face
            display_face_points(frame, landmarks, [0, 68], color='red') # draw face points

            # get position of right eye and display lines
            right_eye_coordinates = get_eye_coordinates(landmarks, [42, 43, 44, 45, 46, 47])
            display_eye_lines(frame, right_eye_coordinates, 'green')

            # define the coordinates of the pupil from the centroid of the right eye
            pupil_coordinates = np.mean([right_eye_coordinates[0], right_eye_coordinates[1]], axis = 0).astype('int')

            if is_blinking(right_eye_coordinates):
                calibration_cut.append(pupil_coordinates)

                # visualize message
                cv2.putText(calibration_page, 'ok',
                            tuple(np.array(corners[corner])-5), cv2.FONT_HERSHEY_SIMPLEX, 2,(255, 255, 255), 5)
                # to avoid is_blinking=True in the next frame
                time.sleep(0.3)
                corner = corner + 1

        print(calibration_cut, '    len: ', len(calibration_cut))
        show_window('projection', calibration_page)
        show_window('frame', cv2.resize(frame,  (640, 360)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # Process Calibration
    x_cut_min, x_cut_max, y_cut_min, y_cut_max = find_cut_limits(calibration_cut)
    
    # Add padding to cut limits to ensure full frame capture
    padding = 50  # Adjust this value as needed
    x_cut_min = max(0, x_cut_min - padding)
    x_cut_max = min(size_screen[1], x_cut_max + padding)
    y_cut_min = max(0, y_cut_min - padding)
    y_cut_max = min(size_screen[0], y_cut_max + padding)
    
    offset_calibrated_cut = [x_cut_min, y_cut_min]

    # Message for user
    print('calibration done. please wait for the keyboard...')
    cv2.putText(calibration_page, 'Calibration Complete. Keyboard Loading...',
                tuple((np.array(size_screen)/4).astype('int')), 
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5)
    show_window('projection', calibration_page)

    cv2.waitKey(3000)  # Wait for 3 seconds
    cv2.destroyAllWindows()

    print('keyboard appearing')

    # Writing Loop
    pressed_key = True
    string_to_write = "text: "
    while(True):
        ret, frame = camera.read()   # Capture frame
        if not ret or frame is None:
            print("Frame capture failed")
            break

        frame = adjust_frame(frame)  # rotate / flip

        # Ensure we don't go out of frame bounds
        cut_frame = np.copy(frame[y_cut_min:y_cut_max, x_cut_min:x_cut_max, :])

        # make & display on frame the keyboard
        keyboard_page = make_black_page(size = size_screen)
        dysplay_keyboard(img = keyboard_page, keys = key_points)
        text_page = make_white_page(size = (200, 800))

        gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # gray-scale to work with

        # detect faces in frame with safe detection
        faces = safe_detect_faces(gray_scale_frame, detector)

        if faces:
            face = faces[0]  # Get the first/largest face
            display_box_around_face(frame, [face.left(), face.top(), face.right(), face.bottom()], 'green', (20, 40))

            landmarks = predictor(gray_scale_frame, face) # find points in face
            display_face_points(frame, landmarks, [0, 68], color='red') # draw face points

            # get position of right eye and display lines
            right_eye_coordinates = get_eye_coordinates(landmarks, [42, 43, 44, 45, 46, 47])
            display_eye_lines(frame, right_eye_coordinates, 'green')

            # define the coordinates of the pupil from the centroid of the right eye
            pupil_on_frame = np.mean([right_eye_coordinates[0], right_eye_coordinates[1]], axis = 0).astype('int')

            # work on the calibrated cut-frame
            pupil_on_cut = np.array([pupil_on_frame[0] - offset_calibrated_cut[0], 
                                      pupil_on_frame[1] - offset_calibrated_cut[1]])
            cv2.circle(cut_frame, (pupil_on_cut[0], pupil_on_cut[1]), 
                       int(take_radius_eye(right_eye_coordinates)/1.5), (255, 0, 0), 3)

            if pupil_on_cut_valid(pupil_on_cut, cut_frame):
                pupil_on_keyboard = project_on_page(img_from = cut_frame[:,:, 0], 
                                                    img_to = keyboard_page[:,:, 0], 
                                                    point = pupil_on_cut)

                # draw circle at pupil_on_keyboard on the keyboard
                cv2.circle(keyboard_page, (pupil_on_keyboard[0], pupil_on_keyboard[1]), 40, (0, 255, 0), 3)

                if is_blinking(right_eye_coordinates):
                    pressed_key = identify_key(key_points = key_points, 
                                               coordinate_X = pupil_on_keyboard[1], 
                                               coordinate_Y = pupil_on_keyboard[0])

                    if pressed_key:
                        if pressed_key=='del':
                            string_to_write = string_to_write[: -1]
                        elif pressed_key != False:
                            string_to_write = string_to_write + pressed_key

                    time.sleep(0.3) # to avoid is_blinking=True in the next frame

        # print on screen the string
        cv2.putText(text_page, string_to_write,
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 0), 5)

        # visualize windows
        show_window('projection', keyboard_page)
        show_window('frame', cv2.resize(frame, (int(frame.shape[1] *resize_frame), int(frame.shape[0] *resize_frame))))
        show_window('cut_frame', cv2.resize(cut_frame, (int(cut_frame.shape[1] *resize_eye_frame), int(cut_frame.shape[0] *resize_eye_frame))))
        show_window('text_page', text_page)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    shut_off(camera) # Shut camera / windows off

if __name__ == "__main__":
    main()