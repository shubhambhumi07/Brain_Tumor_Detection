## 🧠 Brain Tumor Detection with Deep Learning

This is a web application built using **TensorFlow**, **Keras**, and **Streamlit** that allows users to upload **MRI images** and detect the presence of a **brain tumor** using a deep learning model trained on MRI datasets.

---
### ✨ Features

* Upload any MRI image (`.jpg`, `.jpeg`, `.png`)
* Instant prediction with confidence score
* Built using **MobileNetV2** for efficient and fast inference
* Model trained with data augmentation and fine-tuning
* Lightweight and ready for deployment

---

### 🧰 Tech Stack

* Python
* TensorFlow / Keras
* Streamlit
* MobileNetV2 (pretrained on ImageNet)
* PIL (Python Imaging Library)
* NumPy

---

### 📁 Project Structure

```bash
.
├── app.py                          # Streamlit web app
├── train_model.py                 # Training script
├── trained_models/
│   └── brain_tumor_model.h5       # Trained model file
├── datasets/
│   └── combined_dataset_split/    # Dataset (train, val, test)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

### 🔧 Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/brain-tumor-detector.git
   cd brain-tumor-detector
   ```

2. **Create and activate virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download / place your trained model**

   * Ensure `brain_tumor_model.h5` is inside `trained_models/`

5. **Run the app**

   ```bash
   streamlit run app.py
   ```

---

### 🏋️‍♂️ Model Training

If you want to train your own model:

```bash
python train_model.py
```

Make sure your dataset is structured as:

```
datasets/
└── combined_dataset_split/
    ├── train/
    │   ├── tumor/
    │   └── no_tumor/
    ├── val/
    └── test/
```

---

### ✅ Sample Output

```
🧠 Brain Tumor Detected
🧪 Confidence: 94.28%
🧮 Raw model output: 0.9428
```

---

### 📦 Requirements

* TensorFlow >= 2.9
* streamlit
* numpy
* pillow

Create `requirements.txt` with:

```txt
tensorflow
streamlit
numpy
pillow
```

---

### 🌐 Deployment (Optional)

To deploy on **Streamlit Cloud**:

1. Push this project to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your repo and deploy!

---

### 🤝 Acknowledgements

* MRI dataset from [Kaggle](https://www.kaggle.com/datasets)
* MobileNetV2: Pretrained model by Google
* Inspired by medical image classification research

---

### 📬 Contact

**Shubham Singh**
Email: [your-email@example.com](mailto:your-email@example.com)
GitHub: [@yourusername](https://github.com/yourusername)