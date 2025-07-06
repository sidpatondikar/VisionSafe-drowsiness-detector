import cv2
import mediapipe as mp
import numpy as np
import os

class HeadPoseEstimator:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 3D model points for head pose estimation
        self.model_points_3D = np.array([
            (0.0, 0.0, 0.0),            # Nose tip
            (0.0, -330.0, -65.0),       # Chin
            (-225.0, 170.0, -135.0),    # Left eye left corner
            (225.0, 170.0, -135.0),     # Right eye right corner
            (-150.0, -150.0, -125.0),   # Left mouth corner
            (150.0, -150.0, -125.0)     # Right mouth corner
        ])

    def estimate_pose(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return frame, None

        landmarks = results.multi_face_landmarks[0].landmark
        ih, iw = frame.shape[:2]

        try:
            image_points_2D = np.array([
                [landmarks[1].x * iw, landmarks[1].y * ih],     # Nose tip
                [landmarks[152].x * iw, landmarks[152].y * ih], # Chin
                [landmarks[263].x * iw, landmarks[263].y * ih], # Left eye
                [landmarks[33].x * iw, landmarks[33].y * ih],   # Right eye
                [landmarks[287].x * iw, landmarks[287].y * ih], # Left mouth
                [landmarks[57].x * iw, landmarks[57].y * ih]    # Right mouth
            ], dtype="double")
        except IndexError:
            return frame, None

        focal_length = iw
        center = (iw / 2, ih / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))  # No lens distortion

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points_3D, image_points_2D, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return frame, None

        rmat, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rmat, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch = -euler_angles[0].item()
        yaw   =  euler_angles[1].item()
        roll  =  euler_angles[2].item()

        # Draw 3D axis from nose tip
        nose_2d = tuple(image_points_2D[0].astype(int))
        axis_3d = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100]])
        axis_2d, _ = cv2.projectPoints(axis_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        p1 = nose_2d
        p2 = tuple(axis_2d[0].ravel().astype(int))
        p3 = tuple(axis_2d[1].ravel().astype(int))
        p4 = tuple(axis_2d[2].ravel().astype(int))
        cv2.line(frame, p1, p2, (0, 0, 255), 2)
        cv2.line(frame, p1, p3, (0, 255, 0), 2)
        cv2.line(frame, p1, p4, (255, 0, 0), 2)

        # Draw angle text
        # cv2.putText(frame, f"Pitch: {pitch:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        # cv2.putText(frame, f"Yaw: {yaw:.2f}",   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        # cv2.putText(frame, f"Roll: {roll:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # Overlay angles
        # cv2.putText(frame, f"Pitch: {pitch:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        # cv2.putText(frame, f"Yaw: {yaw:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        # cv2.putText(frame, f"Roll: {roll:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        return frame, (pitch, yaw, roll)


def run_realtime_pose_estimation():
    cap = cv2.VideoCapture("data/test_clips/head_pose_demo_3.mp4")
    estimator = HeadPoseEstimator()

    os.makedirs("outputs", exist_ok=True)

    # Use MJPG codec and AVI file for safe output
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('outputs/head_pose_output_3.avi', fourcc, 20.0, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        output_frame, _ = estimator.estimate_pose(frame)

        # Ensure output_frame has expected shape and dtype
        if output_frame.shape[:2] == (480, 640) and output_frame.dtype == 'uint8':
            out.write(output_frame)

    cap.release()
    out.release()
    print("✅ Output saved to: outputs/head_pose_output.avi")



if __name__ == "__main__":
    run_realtime_pose_estimation()
