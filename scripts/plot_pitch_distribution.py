import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

df = pd.read_csv("outputs/nthu_pose_results.csv")
df['label'] = df['label'].astype(str)  # ✅ Ensure categorical labels

print("Labels:", df['label'].unique())  # ✅ Debug: print unique labels

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="pitch",
    hue="label",
    bins=40,
    kde=True,
    palette="muted"
)

plt.title("Pitch Distribution: Drowsy vs Not Drowsy")
plt.xlabel("Pitch (degrees)")
plt.ylabel("Frequency")
plt.legend(title="Label")
plt.tight_layout()

os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/pitch_distribution_by_label.png")
print("✅ Saved: outputs/pitch_distribution_by_label.png")
