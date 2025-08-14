# 🎮 Subway Surfers Gesture Controller

**Control Subway Surfers like a boss — no keyboard, no mouse, just your hands! 🖐️**

This project uses **computer vision + deep learning** to detect your hand gestures via webcam and translate them into in-game actions:

* ✊ Jump
* ✋ Crouch
* 👈 Move Left
* 👉 Move Right

No more arrow keys — just wave like you mean it! 😎

---

## 📸 How It Works

* **BlazePalm** detects your hand in real-time (courtesy of [original BlazePalm repo](https://github.com/vidursatija/BlazePalm))
* Your gestures are interpreted using custom logic
* We send keyboard events to Subway Surfers in your browser or desktop version
* You enjoy running endlessly and smashing your high scores!

---

## 🛠 Tech Stack

* **Python 3.12+**
* **OpenCV** (for webcam input)
* **PyTorch** (for hand detection)
* **PyAutoGUI** (to simulate key presses)
* **NumPy** (because every ML project needs NumPy 😅)

---

## 📦 Installation

```bash
# Clone this repo
git clone https://github.com/Kris0codes/SubwayGestureController.git
cd SubwayGestureController

# Install dependencies
pip install -r requirements.txt
```

## 🪛Requirements

>opencv-python
>numpy
>pyautogui
>torch
>torchvision

---


⚠ **Note:**
This repo **does not** include model weights (`palmdetector.pth`, `anchors.npy`) to avoid large file uploads.
👉 You can get them from the **[original BlazePalm repo](https://github.com/vidursatija/BlazePalm)** and place them inside the `ML/` directory.

---

## ▶️ Usage

1. Open **Subway Surfers** (browser or desktop).
2. Run the script:

   ```bash
   python subway_gesture_controller.py
   ```
3. Make sure the camera is on, wave your hand, and… **become the subway ninja** 🥷

---

## 🎯 Gestures

| Gesture            | Action     |
| ------------------ | ---------- |
| ✊ Closed Fist      | Jump       |
| ✋ Open Palm        | Crouch     |
| 👈 Tilt Hand Left  | Move Left  |
| 👉 Tilt Hand Right | Move Right |

---

## 💡 Pro Tips

* Wear contrasting sleeves to make detection easier.
* Don’t gesture too close to the camera — you’re not trying to boop it.
* Your cat might accidentally play the game if it walks past the camera. 🐈

---

## 📜 License

This project is released under the MIT License.
BlazePalm original code belongs to its respective authors — full credit to them.

---

