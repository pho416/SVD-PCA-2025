# Matrix Factorization & Dimensionality Reduction: SVD vs. PCA

**Course:** Mathematics for Computer Science  
**Institution:** University of Information Technology (UIT)  

---

## 📖 Project Overview

This project explores the mathematical foundations and practical applications of **Dimensionality Reduction** in Computer Vision. We implement and compare two powerful Linear Algebra techniques for image compression and restoration:

1.  **Singular Value Decomposition (SVD):** A fundamental algebraic factorization method implemented **entirely from scratch** to demonstrate understanding of the underlying mechanics (Power Iteration, Gram-Schmidt process).
2.  **Principal Component Analysis (PCA):** A statistical approach implemented using **scikit-learn**, optimized for full-color (RGB) image compression with production-grade engineering metrics.

## 🚀 Key Features

### 1. SVD Implementation (The "Math" Approach)
* **Core Engine:** Built from ground zero without relying on `numpy.linalg` for the decomposition logic.
* **Algorithms:** Implements Power Iteration for eigenvalue search and Gram-Schmidt for orthogonalization.
* **Application:** Grayscale image compression via Low-Rank Approximation ($A_k = U_k \Sigma_k V_k^T$).

### 2. PCA Implementation (The "Engineering" Approach)
* **Multi-Channel Support:** Handles full **RGB** color images by decomposing and compressing channels independently.
* **Advanced Metrics:** Automatically calculates:
    * **Compression Ratio:** Theoretical storage savings.
    * **Fidelity:** Mathematical accuracy ($1 - RelativeError$).
    * **Visual Quality:** Qualitative assessment (Excellent/Good/Poor).
* **Visualization:** Side-by-side comparison of original vs. reconstructed images at various $k$ levels.

## Team Members

- 24521824 - Bùi Cao Trí
- 24521878 - Hồ Quốc Trung
- 24521743 - Phan Công Thuận
- 24521779 - Phạm Tiến
---

## 📂 Project Structure

```text
Final_Project/
│
├── src/
│   └── svd.py                  # Core mathematical module (SVD algorithm from scratch)
│
├── notebooks/
│   ├── SVD_Demo.ipynb          # Implementation Demo (SVD)
│   └── PCA_Demo.ipynb          # Implementation Demo (PCA)
│   └── SVD_Image_Compression.ipynb   # Demo: Grayscale compression using custom SVD
│   └── PCA_Image_Compression.ipynb   # Demo: RGB compression using sklearn PCA
|   └── sample_image.jpg
│   
│
└── README.md                   # Project documentation
