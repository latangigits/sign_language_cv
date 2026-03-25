# 🤟 Sign Language Gesture Recognition

## About this project

This project is a simple **sign language gesture recognition system** built using Python and computer vision.
The idea was to create something that can recognize basic hand gestures through a webcam in real time.

I trained a model on a custom dataset of gesture images and used it to predict gestures live using OpenCV.

---

## What it does

* Uses your webcam to capture hand gestures
* Predicts the gesture using a trained model
* Displays the result in real time

---

## Tech used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy

---

## Project files

```
dataset/              → images used for training  
gesture_model.h5      → trained model  
train.py              → script to train the model  
predict.py            → script for real-time prediction  
demo video .MP4       → sample output  
```

---

## How to run

### 1. Clone the repo

```
git clone https://github.com/latangigits/sign_language_cv.git
cd sign_language_cv
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Train the model (optional)

```
python3 train.py
```

### 4. Run the project

```
python3 predict.py
```

---

## Demo

You can check the demo video in the repository to see how it works.

---

## Notes

* Make sure camera permission is enabled
* Good lighting helps improve accuracy
* If the model file is missing, run `train.py` first

---

## Why I made this

I wanted to explore how computer vision and deep learning can be used for something practical like gesture recognition. This project helped me understand how models are trained and used in real-time applications.

---

## Future improvements

* Add more gesture classes
* Improve accuracy
* Build a simple UI
* Deploy it as a web app

---

## Author

Lathangi
