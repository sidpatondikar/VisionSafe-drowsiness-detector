import pandas as pd
import os

# Load CSV
input_path = "outputs/nthu_pose_results.csv"
output_path = "outputs/nthu_pose_results_labeled.csv"

df = pd.read_csv(input_path)

# Set threshold for posture-based drowsiness
PITCH_THRESHOLD = -10  # degrees

# Add posture_drowsy column
df['posture_drowsy'] = df['pitch'] < PITCH_THRESHOLD

# Save updated CSV
os.makedirs("outputs", exist_ok=True)
df.to_csv(output_path, index=False)
print(f"✅ Saved labeled CSV with posture_drowsy column to {output_path}")
