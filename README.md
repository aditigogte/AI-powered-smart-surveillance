# GuardianAI: Smart Surveillance & Suspicious Activity Detection System

GuardianAI is an edge-AI computer vision solution designed to automate real-time video analytics for security operations. Powered by YOLOv8 and OpenCV, the system processes video feeds to track individuals, detect behavioral anomalies, and flag security breaches instantly with visual alerts, localized audio, and photographic evidence logging.

---

## ✨ Core Features

The system actively monitors video streams and executes **7 core behavioral and environmental analytics modules**:

1. **Automatic Night Vision Exposure:** Dynamically evaluates frame luminosity and applies adaptive exposure scaling ($\alpha=1.5$, $\beta=30$) to low-light feeds.
2. **Real-Time Bounding & Centroid Tracking:** Tracks individuals uniquely using Euclidean distance tracking logic across frame buffers.
3. **Restricted Zone Intrusion Detection:** Monitors a custom digital perimeter (Spatiotemporal bounding box) and flags immediate unauthorized breaches.
4. **Running & Pacing Detection:** Measures sudden velocity spikes in human trajectories against a set pixels-per-frame threshold.
5. **Loitering Analytics:** Tracks individual spatiotemporal paths to flag subjects lingering in a localized area with negligible cumulative movement over time.
6. **Proximity & Fight Matrix Assessment:** Dynamically measures the distance matrix between all detected individuals to identify physical proximity thresholds indicative of altercations.
7. **Secondary Weapon Scanning:** Integrates an optional custom YOLO architecture layer to parse frames for firearms or dangerous objects.

---

## ⚙️ System Architecture & Workflow