# Enhancing Image Classification Accuracy Through Noise Reduction (MNIST)

This repository investigates how different types of image noise affect digit classification and evaluates denoising strategies to restore accuracy. The project uses the MNIST dataset with added Gaussian, salt-and-pepper, speckle, and mixed noise, and applies classical filters and a convolutional autoencoder before classification with a CNN.

---

## Features
- Noise injection: Gaussian, salt-and-pepper, speckle, and a mixed-noise scenario.
- Baseline CNN classifier trained on clean MNIST.
- Denoising approaches: Median, Gaussian, Bilateral, Non-local Means, and a convolutional Autoencoder.
- Hybrid approaches combining classical filters with deep learning.
- Accuracy evaluation and visualizations of noisy vs. denoised digits.

---

## Repository Structure
```
.
├─ executable_code.py      # Main script: noise generation, CNN training, denoising, evaluation, visualization
└─ README.md               # Project documentation
```

---

## Example Results
- Clean test accuracy: **~98.6%**
- Noisy test accuracy: **~70.0%**
- Gaussian filter (best individual filter on mixed noise): **~91.6%**
- Hybrid methods (e.g., Bilateral → Autoencoder): **~82.7%**

Note: Results vary slightly depending on environment and random seeds.

---

## Requirements
- Python 3.9+
- Install dependencies:
  ```bash
  pip install numpy pandas matplotlib opencv-python tensorflow keras
  ```

---

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/username/Enhancing-Image-Classification-Accuracy-Through-Noisy-Reduction.git
   cd Enhancing-Image-Classification-Accuracy-Through-Noisy-Reduction
   ```

2. Run the script:
   ```bash
   python executable_code.py
   ```

---

## What Happens When You Run It
1. Loads MNIST dataset and prepares clean baseline.
2. Generates noisy versions of test images.
3. Builds and trains a CNN classifier on clean data.
4. Evaluates CNN performance on clean, noisy, and denoised datasets.
5. Saves output images (clean, noisy, and denoised samples) and prints accuracy tables.

---

## Output Files
- Images are saved by default to:
  ```
  /home/sk/Desktop/
  ```
  You can change this path in the `save_images(...)` function inside `executable_code.py`.

For example:
```python
plt.savefig("outputs/mixed_noise_data.png")
```
and then create the folder:
```bash
mkdir outputs
```

---

## Troubleshooting
- **TensorFlow/Keras errors**: Ensure both are installed with compatible versions (`pip install tensorflow keras`).
- **OpenCV missing**: Install via `pip install opencv-python`.
- **Images not saving**: Make sure the directory exists and update the path in the script if needed.

---


