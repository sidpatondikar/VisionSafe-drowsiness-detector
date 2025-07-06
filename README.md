### Web Application - https://visionsafe-web-app.vercel.app/

# 🚗 VisionSafe – Real-Time Driver Alertness & Distraction Monitoring System

VisionSafe is a real-time computer vision system that detects drowsiness and distraction in drivers using a standard webcam. It leverages deep learning, facial landmark tracking, and head pose estimation to continuously assess a driver's alertness and provide timely audio/visual warnings — all running efficiently on consumer-grade hardware.

---

## 🎯 Project Objective

To build a lightweight, real-time monitoring system that detects:
- 👁️ **Eye State** (open vs closed)
- 🧭 **Head Pose Orientation** (pitch, yaw, roll)
- 💤 **Drowsiness** and 📵 **Distraction**
- 🔊 **Live Alerts** with fatigue scoring

---

## ⚙️ Core Features

- **CNN-Based Eye State Classifier**  
  Uses a fine-tuned MobileNetV2 to classify eye state from webcam frames.

- **Head Pose Estimation with MediaPipe + OpenCV**  
  Estimates pitch/yaw/roll from facial landmarks using `cv2.solvePnP`.

- **Real-Time Video Monitoring**  
  Streams webcam input and overlays predictions, angles, and scores.

- **Distraction Score (0–100)**  
  Frame-by-frame logic that accumulates risk score based on eye closure and head orientation.

- **Custom Audio Alerts**  
  Distinct sounds for drowsiness vs distraction with cooldown timers to avoid alert fatigue.


---

## How It Works

1. Captures webcam frames with OpenCV.
2. Detects facial landmarks with MediaPipe.
3. Crops and classifies each eye using a trained CNN.
4. Estimates head pose to detect looking away or nodding.
5. Classifies current status: `Safe`, `Drowsy`, or `Distracted`.
6. Maintains a distraction score and triggers alerts when thresholds are crossed.

---

## 📁 Project Structure

``` 
.
├── scripts/
|   ├── realtime/                    # Real-time monitoring + webcam pipeline
│   |   ├── camera_pipeline.py
│   |
|   ├── pose_estimation/            # Head pose-related logic and analysis
│   |   ├── head_pose_estimation.py
│   |   ├── head_pose_nthu_test.py
│   |   ├── add_posture_flag.py
│   |   ├── plot_pitch_distribution.py
│   |
|   ├── model_training/             # CNN model training and experimentation
│   |   ├── train_eye_notebook.ipynb
│   |
|   ├── preprocessing/              # Dataset preparation and parsing
│       ├── preprocess_mrl.py
│       ├── preprocess_nthu.py
|
├── models/
|   ├── eye_state_model.keras        # Trained CNN eye state model
|   ├── eye_state_model
|
├──assets/
|   ├── alert_distracted.wav         # Alert sound for distracted status
|   ├── alert_drowsy.wav             # Alert sound for drowsy status
|
├── requirements.txt
└── README.md

```
---

## 📂 Datasets Used

- **👁️ MRL Eye Dataset**  
  Used to train the CNN-based eye-state classifier.  
  🔗 [https://mrl.cs.vsb.cz/eyedataset](https://mrl.cs.vsb.cz/eyedataset)

- **🧑‍💻 NTHU-DDD2 Dataset**  
  Used to simulate and test driver drowsiness in real-world conditions.  
  🔗 [https://www.kaggle.com/datasets/banudeep/nthuddd2](https://www.kaggle.com/datasets/banudeep/nthuddd2)

---

## 🧪 Model Performance

| Metric     | Value     |
|------------|-----------|
| Accuracy   | 92%       |
| AUC        | 0.98+     |
| Recall (Drowsy) | 99%   |
| Precision (Awake) | 99% |

---

## 📌 Packages Used

**Core** - `numpy`, `pandas`, `opencv-python`, `pillow`, `tqdm`, `pyyaml`

**ML/DL** - `tensorflow`, `scikit-learn`, `matplotlib`, `seaborn`

**Facial landmarks** - `mediapipe`

**Alerts** - `pygame`

---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Start the real-time detection pipeline
python3 scripts/realtime/camera_pipeline.py
```

---

## 👨‍💻 Author

Developed by `Siddharth Patondikar`, as a real-world showcase project combining computer vision, deep learning, and real-time systems.

---

## 📣 Contact & Showcase

- 🔗 Portfolio: [[sidpatondikar.web](https://sidpatondikar-web.vercel.app/)]
- 🧠 Resume Project: **VisionSafe**
- 📫 Contact: **siddharth.patondikar@gmail.com**

---

> VisionSafe demonstrates how real-time deep learning can be used to solve critical safety problems with accessible hardware. It’s fast, modular, and open for extension (e.g., yawning detection, temporal smoothing, Streamlit/FastAPI web apps).
