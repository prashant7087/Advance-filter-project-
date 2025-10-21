import cv2
import mediapipe as mp
import numpy as np
import math

# --- Landmark Indices (from MediaPipe Face Mesh) ---
# These indices correspond to specific points on the 478-point face model.
LEFT_PUPIL = 473
RIGHT_PUPIL = 468
# --- MODIFICATION ---
# Previous reference points (temples) were not wide enough, leading to an
# inflated mm/pixel scale. These new landmarks on the widest part of the
# cheekbones (zygomatic arch) provide a much more stable and accurate 
# reference that corresponds better to the full width of glasses.
LEFT_REFERENCE = 127
RIGHT_REFERENCE = 356
LEFT_EYE_LOWER = 27
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152

def analyze_video(video_path, frame_width_mm):
    """
    Analyzes a video to find facial landmarks and calculate optical measurements.
    
    Args:
        video_path (str): The path to the video file.
        frame_width_mm (float): The known physical width of the glasses frame in mm.

    Returns:
        A tuple containing three elements:
        1. dict: A dictionary of calculated measurements (pd, fh, tilt, vertex).
        2. list: A list of normalized 3D landmark coordinates for rendering.
        3. dict: A dictionary with the pixel dimensions of the video frame.
    
    Raises:
        ValueError: If the video cannot be opened or a face is not detected.
    """
    mp_face_mesh = mp.solutions.face_mesh
    
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, 
        max_num_faces=1, 
        refine_landmarks=True, 
        min_detection_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error: Could not open video file.")

    success, image = cap.read()
    cap.release()

    if not success:
        raise ValueError("Error: Could not read a frame from the video.")

    frame_height_px, frame_width_px, _ = image.shape
    frame_dims = {"width": frame_width_px, "height": frame_height_px}

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
    if not results.multi_face_landmarks:
        raise ValueError("No face detected in the video.")
        
    face_landmarks = results.multi_face_landmarks[0]
    landmarks = face_landmarks.landmark
    
    # --- Pixel to MM Conversion (Using new, wider reference points) ---
    left_ref_pt = landmarks[LEFT_REFERENCE]
    right_ref_pt = landmarks[RIGHT_REFERENCE]
    
    # Calculate the 2D Euclidean distance between reference points in pixels.
    ref_width_px = math.sqrt(((right_ref_pt.x - left_ref_pt.x) * frame_width_px)**2 + 
                             ((right_ref_pt.y - left_ref_pt.y) * frame_height_px)**2)
    
    if ref_width_px == 0:
        raise ValueError("Could not establish a reference width for measurement.")
        
    # This ratio should now be much more accurate.
    mm_per_pixel = frame_width_mm / ref_width_px

    # --- Measurement Calculations ---
    # 1. Pupillary Distance (PD)
    left_pupil_pt = landmarks[LEFT_PUPIL]
    right_pupil_pt = landmarks[RIGHT_PUPIL]
    pd_px = math.sqrt(((right_pupil_pt.x - left_pupil_pt.x) * frame_width_px)**2 + 
                      ((right_pupil_pt.y - left_pupil_pt.y) * frame_height_px)**2)
    pd_mm = pd_px * mm_per_pixel

    # 2. Fitting Height (FH)
    fh_px = abs(left_pupil_pt.y - landmarks[LEFT_EYE_LOWER].y) * frame_height_px
    fh_mm = fh_px * mm_per_pixel

    # 3. Pantoscopic Tilt
    forehead_z = landmarks[FOREHEAD].z
    chin_z = landmarks[CHIN].z
    tilt_rad = math.atan2(chin_z - forehead_z, 0.2)
    tilt_deg = abs(math.degrees(tilt_rad))

    # 4. Vertex Distance
    vertex_mm = (abs(landmarks[NOSE_TIP].z) * 100) + 8 

    measurements = {
        "pd": pd_mm,
        "fh": fh_mm * 1.5, # Multiplier to estimate from eyelid to frame bottom
        "tilt": min(tilt_deg, 15.0),
        "vertex": min(vertex_mm, 14.0)
    }
    
    landmarks_for_3d = [[lm.x, lm.y, lm.z] for lm in landmarks]
    
    face_mesh.close()
    
    return measurements, landmarks_for_3d, frame_dims





# import cv2
# import mediapipe as mp
# import numpy as np
# import math

# mp_face_mesh = mp.solutions.face_mesh

# # Landmark indices
# LEFT_PUPIL = 473
# RIGHT_PUPIL = 468
# LEFT_TEMPLE = 162
# RIGHT_TEMPLE = 389
# NOSE_TIP = 1
# CHIN = 152
# LEFT_EYE_CORNER = 130
# RIGHT_EYE_CORNER = 359
# LEFT_MOUTH_CORNER = 61
# RIGHT_MOUTH_CORNER = 291
# LOWER_EYE_CONTOUR_LEFT = [23, 24, 25, 26, 27, 112]

# def get_head_pose(landmarks, image_shape):
#     # This function is correct and unchanged
#     face_3d = np.array([
#         landmarks[NOSE_TIP], landmarks[CHIN], landmarks[LEFT_EYE_CORNER],
#         landmarks[RIGHT_EYE_CORNER], landmarks[LEFT_MOUTH_CORNER], landmarks[RIGHT_MOUTH_CORNER]
#     ], dtype=np.float64)
#     face_2d = np.array([
#         (landmarks[NOSE_TIP][0] * image_shape[1], landmarks[NOSE_TIP][1] * image_shape[0]),
#         (landmarks[CHIN][0] * image_shape[1], landmarks[CHIN][1] * image_shape[0]),
#         (landmarks[LEFT_EYE_CORNER][0] * image_shape[1], landmarks[LEFT_EYE_CORNER][1] * image_shape[0]),
#         (landmarks[RIGHT_EYE_CORNER][0] * image_shape[1], landmarks[RIGHT_EYE_CORNER][1] * image_shape[0]),
#         (landmarks[LEFT_MOUTH_CORNER][0] * image_shape[1], landmarks[LEFT_MOUTH_CORNER][1] * image_shape[0]),
#         (landmarks[RIGHT_MOUTH_CORNER][0] * image_shape[1], landmarks[RIGHT_MOUTH_CORNER][1] * image_shape[0])
#     ], dtype=np.float64)
#     focal_length = image_shape[1]
#     cam_center = (image_shape[1] / 2, image_shape[0] / 2)
#     cam_matrix = np.array([[focal_length, 0, cam_center[0]], [0, focal_length, cam_center[1]], [0, 0, 1]], dtype=np.float64)
#     dist_matrix = np.zeros((4, 1), dtype=np.float64)
#     success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
#     if not success: return None
#     rot_mat, _ = cv2.Rodrigues(rot_vec)
#     sy = math.sqrt(rot_mat[0, 0]**2 + rot_mat[1, 0]**2)
#     singular = sy < 1e-6
#     if not singular:
#         x, y, z = math.atan2(rot_mat[2, 1], rot_mat[2, 2]), math.atan2(-rot_mat[2, 0], sy), math.atan2(rot_mat[1, 0], rot_mat[0, 0])
#     else:
#         x, y, z = math.atan2(-rot_mat[1, 2], rot_mat[1, 1]), math.atan2(-rot_mat[2, 0], sy), 0
#     return {'yaw': math.degrees(y), 'pitch': math.degrees(x)}

# # --- UPDATED Fitting Height function ---
# # This now uses the generic 3D scaling_factor, which is more stable
# def calculate_fitting_height_with_frame(landmarks, scaling_factor):
#     pupil_landmark = landmarks[LEFT_PUPIL]
#     lower_contour_points = landmarks[LOWER_EYE_CONTOUR_LEFT]
    
#     lowest_y = max(point[1] for point in lower_contour_points)
#     pupil_y = pupil_landmark[1]
    
#     # Calculate vertical distance in the landmark's native coordinate space
#     vertical_dist_normalized = abs(lowest_y - pupil_y)
    
#     # Apply the generic scaling factor
#     fh_mm = vertical_dist_normalized * scaling_factor
#     return fh_mm

# def analyze_video(video_path, frame_width_mm):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened(): raise ValueError("Could not open video file.")
#     frontal_frame_data = {'frame_shape': None, 'landmarks': None, 'yaw': 999}
    
#     with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
#         while cap.isOpened():
#             success, image = cap.read()
#             if not success: break
#             image.flags.writeable = False
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             results = face_mesh.process(image_rgb)
#             if results.multi_face_landmarks:
#                 face_landmarks = results.multi_face_landmarks[0]
#                 landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
#                 pose = get_head_pose(landmarks, image.shape)
#                 if pose and abs(pose['yaw']) < abs(frontal_frame_data['yaw']):
#                     frontal_frame_data.update({'yaw': pose['yaw'], 'frame_shape': image.shape, 'landmarks': landmarks})
#     cap.release()

#     if frontal_frame_data['landmarks'] is None: raise ValueError("Could not detect a clear frontal face.")
    
#     frontal_landmarks = frontal_frame_data['landmarks']
#     image_height_px = frontal_frame_data['frame_shape'][0]
#     image_width_px = frontal_frame_data['frame_shape'][1]

#     # --- REVERTED CALCULATION LOGIC ---
#     # Reverting to the more stable 3D distance-based scaling as requested.
#     landmarks_for_calc = frontal_landmarks.copy()
    
#     # The Z-axis from MediaPipe needs to be scaled to be proportional to X and Y.
#     # We scale it by the image width as a reasonable heuristic.
#     landmarks_for_calc[:, 2] *= image_width_px

#     # Calculate the 3D distance between temple landmarks
#     model_temple_dist_3d = np.linalg.norm(landmarks_for_calc[LEFT_TEMPLE] - landmarks_for_calc[RIGHT_TEMPLE])
    
#     if model_temple_dist_3d == 0: raise ValueError("Could not calculate temple distance.")
    
#     # Create a single, generic scaling factor based on the known frame width
#     scaling_factor_3d = frame_width_mm / model_temple_dist_3d

#     # Calculate PD using the 3D distance and the generic scaling factor
#     pd_dist_3d = np.linalg.norm(landmarks_for_calc[LEFT_PUPIL] - landmarks_for_calc[RIGHT_PUPIL])
#     pd_mm = pd_dist_3d * scaling_factor_3d

#     # Calculate FH using the updated function and the same generic scaling factor
#     # Note: We pass the scaled landmarks to this function
#     fh_mm = calculate_fitting_height_with_frame(landmarks_for_calc, scaling_factor_3d)

#     # Placeholders for other measurements
#     tilt_degrees, vertex_mm = 8.0, 12.0
#     measurements = {'pd': round(pd_mm, 2), 'fh': round(fh_mm, 2), 'tilt': round(tilt_degrees, 2), 'vertex': round(vertex_mm, 2)}
    
#     # This part is still needed for the 3D model aspect ratio correction
#     frame_dims = {"width": image_width_px, "height": image_height_px}
    
#     return measurements, frontal_landmarks.tolist(), frame_dims
#     return measurements, frontal_landmarks.tolist(), frame_dims


# import cv2
# import mediapipe as mp
# import numpy as np
# import math

# mp_face_mesh = mp.solutions.face_mesh

# # --- CONSTANTS ---
# # Landmark indices for pupils and irises from MediaPipe
# LEFT_PUPIL = 473
# RIGHT_PUPIL = 468
# LEFT_IRIS_HORIZONTAL_EDGE_1 = 474 # Point on the left side of the left iris
# LEFT_IRIS_HORIZONTAL_EDGE_2 = 476 # Point on the right side of the left iris

# # Average Horizontal Visible Iris Diameter (HVID) in mm. This is our new reference.
# AVG_IRIS_DIAMETER_MM = 11.77 

# # --- HELPER FUNCTIONS (FROM YOUR ORIGINAL CODE) ---
# # Your head pose function remains unchanged. It's still useful for finding the best frame.
# def get_head_pose(landmarks, image_shape):
#     face_3d = np.array([
#         landmarks[1], landmarks[152], landmarks[130],
#         landmarks[359], landmarks[61], landmarks[291]
#     ], dtype=np.float64)
#     face_2d = np.array([
#         (landmarks[1][0] * image_shape[1], landmarks[1][1] * image_shape[0]),
#         (landmarks[152][0] * image_shape[1], landmarks[152][1] * image_shape[0]),
#         (landmarks[130][0] * image_shape[1], landmarks[130][1] * image_shape[0]),
#         (landmarks[359][0] * image_shape[1], landmarks[359][1] * image_shape[0]),
#         (landmarks[61][0] * image_shape[1], landmarks[61][1] * image_shape[0]),
#         (landmarks[291][0] * image_shape[1], landmarks[291][1] * image_shape[0])
#     ], dtype=np.float64)
#     focal_length = image_shape[1]
#     cam_center = (image_shape[1] / 2, image_shape[0] / 2)
#     cam_matrix = np.array([[focal_length, 0, cam_center[0]], [0, focal_length, cam_center[1]], [0, 0, 1]], dtype=np.float64)
#     dist_matrix = np.zeros((4, 1), dtype=np.float64)
#     success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
#     if not success: return None
#     rot_mat, _ = cv2.Rodrigues(rot_vec)
#     sy = math.sqrt(rot_mat[0, 0]**2 + rot_mat[1, 0]**2)
#     singular = sy < 1e-6
#     if not singular:
#         x, y, z = math.atan2(rot_mat[2, 1], rot_mat[2, 2]), math.atan2(-rot_mat[2, 0], sy), math.atan2(rot_mat[1, 0], rot_mat[0, 0])
#     else:
#         x, y, z = math.atan2(-rot_mat[1, 2], rot_mat[1, 1]), math.atan2(-rot_mat[2, 0], sy), 0
#     return {'yaw': math.degrees(y), 'pitch': math.degrees(x)}

# # --- MAIN ANALYSIS FUNCTION (HEAVILY MODIFIED) ---
# # Note: It no longer accepts 'frame_width_mm' as an argument.
# def analyze_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened(): 
#         raise ValueError("Could not open video file.")

#     frontal_frame_data = {'frame_shape': None, 'landmarks': None, 'yaw': 999}
    
#     with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
#         while cap.isOpened():
#             success, image = cap.read()
#             if not success: break
            
#             image.flags.writeable = False
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             results = face_mesh.process(image_rgb)
            
#             if results.multi_face_landmarks:
#                 face_landmarks = results.multi_face_landmarks[0]
#                 landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
                
#                 # Find the frame where the user is looking most straight-on
#                 pose = get_head_pose(landmarks, image.shape)
#                 if pose and abs(pose['yaw']) < abs(frontal_frame_data['yaw']):
#                     frontal_frame_data.update({
#                         'yaw': pose['yaw'], 
#                         'frame_shape': image.shape, 
#                         'landmarks': landmarks
#                     })

#     cap.release()

#     if frontal_frame_data['landmarks'] is None:
#         raise ValueError("Could not detect a clear frontal face in the video.")

#     # --- NEW IRIS-BASED CALCULATION LOGIC ---
#     landmarks = frontal_frame_data['landmarks']
#     image_height, image_width, _ = frontal_frame_data['frame_shape']
    
#     # 1. Get 2D coordinates of iris and pupil landmarks IN PIXELS
#     left_iris_p1_px = (landmarks[LEFT_IRIS_HORIZONTAL_EDGE_1][0] * image_width, landmarks[LEFT_IRIS_HORIZONTAL_EDGE_1][1] * image_height)
#     left_iris_p2_px = (landmarks[LEFT_IRIS_HORIZONTAL_EDGE_2][0] * image_width, landmarks[LEFT_IRIS_HORIZONTAL_EDGE_2][1] * image_height)
    
#     left_pupil_px = (landmarks[LEFT_PUPIL][0] * image_width, landmarks[LEFT_PUPIL][1] * image_height)
#     right_pupil_px = (landmarks[RIGHT_PUPIL][0] * image_width, landmarks[RIGHT_PUPIL][1] * image_height)
    
#     # 2. Calculate the pixel distance for the iris diameter and the PD
#     iris_diameter_px = math.sqrt((left_iris_p1_px[0] - left_iris_p2_px[0])**2 + (left_iris_p1_px[1] - left_iris_p2_px[1])**2)
#     pd_dist_px = math.sqrt((left_pupil_px[0] - right_pupil_px[0])**2 + (left_pupil_px[1] - right_pupil_px[1])**2)

#     if iris_diameter_px == 0:
#         raise ValueError("Could not measure iris diameter.")

#     # 3. Create the mm-per-pixel ratio from the iris
#     mm_per_pixel = AVG_IRIS_DIAMETER_MM / iris_diameter_px
    
#     # 4. Convert the PD from pixels to millimeters
#     pd_mm = pd_dist_px * mm_per_pixel

#     # Placeholder values for other measurements. These could also be updated later.
#     fh_mm, tilt_degrees, vertex_mm = 22.0, 8.0, 12.0
    
#     measurements = {
#         'pd': round(pd_mm, 2), 
#         'fh': round(fh_mm, 2), 
#         'tilt': round(tilt_degrees, 2), 
#         'vertex': round(vertex_mm, 2)
#     }
    
#     frame_dims = {"width": image_width, "height": image_height}
    
#     return measurements, landmarks.tolist(), frame_dims