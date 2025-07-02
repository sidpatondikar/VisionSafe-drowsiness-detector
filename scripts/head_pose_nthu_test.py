import pandas as pd
import cv2
import os
from tqdm import tqdm
from head_pose_estimation import HeadPoseEstimator

def load_metadata(csv_path='data/nthu/nthu_metadata.csv'):
    df = pd.read_csv(csv_path)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)  # ✅ Shuffle

def test_head_pose_on_nthu(df, base_dir='data/nthu/', visualize=False):
    estimator = HeadPoseEstimator()
    os.makedirs("outputs", exist_ok=True)

    results = []
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('outputs/nthu_pose_test.avi', fourcc, 20.0, (640, 480))

    MAX_FRAMES = 3000

    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        if idx >= MAX_FRAMES:
            break

        label = row['label']  # 0 = notdrowsy, 1 = drowsy
        subdir = 'drowsy' if label == 1 else 'notdrowsy'
        filename = row['filename']
        image_path = os.path.join(base_dir, subdir, filename)

        if not os.path.exists(image_path):
            continue

        frame = cv2.imread(image_path)
        if frame is None:
            continue

        frame = cv2.resize(frame, (640, 480))
        output_frame, angles = estimator.estimate_pose(frame)

        if angles:
            pitch, yaw, roll = angles

            # Add posture label
            posture_label = "Head Down" if pitch < -10 else "Upright"
            color = (0, 0, 255) if pitch < -10 else (0, 255, 0)
            cv2.putText(output_frame, posture_label, (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            results.append({
                'filename': filename,
                'label': 'drowsy' if label == 1 else 'notdrowsy',
                'pitch': pitch,
                'yaw': yaw,
                'roll': roll
            })

            out.write(output_frame)
            if visualize:
                cv2.imshow("NTHU Head Pose", output_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    out.release()
    if visualize:
        cv2.destroyAllWindows()

    return pd.DataFrame(results)

if __name__ == "__main__":
    df = load_metadata()
    results = test_head_pose_on_nthu(df, visualize=False)

    results.to_csv("outputs/nthu_pose_results.csv", index=False)
    print("✅ Saved video: outputs/nthu_pose_test.avi")
    print("✅ Saved angles: outputs/nthu_pose_results.csv")

    # Quick angle stats
    print("\n📊 Average Pitch:")
    print("  Not Drowsy:", results[results['label'] == 'notdrowsy']['pitch'].mean())
    print("  Drowsy    :", results[results['label'] == 'drowsy']['pitch'].mean())
