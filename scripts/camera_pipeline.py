import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from head_pose_estimation import HeadPoseEstimator

# Load trained eye-state model
eye_model = tf.keras.models.load_model('models/eye_state_model.keras')

# MediaPipe FaceMesh for eye landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Use same indices as head pose code for eyes (based on your MediaPipe mapping)
LEFT_EYE_IDXS = [362, 263]
RIGHT_EYE_IDXS = [33, 133]

pose_estimator = HeadPoseEstimator()

def crop_eye(img, landmarks, indices):
    h, w = img.shape[:2]
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    x1, y1 = pts[0]
    x2, y2 = pts[1]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    size = max(abs(x2 - x1), abs(y2 - y1)) * 2
    x1_crop, y1_crop = max(cx - size // 2, 0), max(cy - size // 2, 0)
    x2_crop, y2_crop = cx + size // 2, cy + size // 2
    eye_img = img[y1_crop:y2_crop, x1_crop:x2_crop]
    if eye_img.size == 0:
        return None
    eye_img = cv2.resize(eye_img, (64, 64))
    eye_img = eye_img.astype(np.float32) / 255.0
    return eye_img[np.newaxis, ...]

def get_eye_state(patch):
    if patch is None:
        return "Unknown"
    pred = eye_model.predict(patch, verbose=0)
    return "Open" if pred[0][0] > 0.5 else "Closed"

def get_driver_status(left, right, pitch):
    if left == right == "Closed":
        return "Drowsy"
    elif abs(pitch) > 20:
        return "Distracted"
    elif left == "Unknown" or right == "Unknown":
        return "Uncertain"
    return "Safe"

def draw_info(frame, left_eye, right_eye, angles, status):
    pitch, yaw, roll = angles
    cv2.putText(frame, f"Left Eye: {left_eye}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.putText(frame, f"Right Eye: {right_eye}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(frame, f"Yaw: {yaw:.1f}",   (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(frame, f"Roll: {roll:.1f}", (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    color = (0,255,0) if status=="Safe" else (0,255,255) if status=="Distracted" else (0,0,255)
    cv2.putText(frame, f"Status: {status}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get head pose estimate
        pose_frame, angles = pose_estimator.estimate_pose(frame.copy())

        if angles:
            pitch, yaw, roll = angles

            # For eyes: extract from landmarks
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark
                left_patch = crop_eye(rgb, landmarks, LEFT_EYE_IDXS)
                right_patch = crop_eye(rgb, landmarks, RIGHT_EYE_IDXS)
                left_state = get_eye_state(left_patch)
                right_state = get_eye_state(right_patch)
                status = get_driver_status(left_state, right_state, pitch)
                draw_info(pose_frame, left_state, right_state, angles, status)
            else:
                cv2.putText(pose_frame, "Face not detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            cv2.imshow("VisionSafe", pose_frame)
        else:
            cv2.imshow("VisionSafe", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
