# SAR Image Colorization with UNet2DModel

![Project Banner](assets/images/banner_image.png)

## 📖 Overview

This repository contains the implementation of a **Synthetic Aperture Radar (SAR) image colorization model** using a **UNet2DModel**. The model transforms grayscale SAR images into their optical (colorized) counterparts, enhancing the interpretability of SAR data for various applications such as remote sensing and environmental monitoring.

<p align="center">
  <img src="assets/images/comparison_image.png" alt="SAR to Optical Image Comparison" width="600">
</p>

---

## 🧰 Features

- **Encoder-Decoder Architecture**: Built using the flexible UNet2DModel.
- **Attention Mechanisms**: Incorporated to improve the quality of colorization.
- **Custom Dataset Support**: Compatible with paired SAR and optical image datasets.
- **High Performance**: Optimized for efficient GPU usage and minimal memory consumption.
- **Training Logs**: Includes visualizations for loss, PSNR, and SSIM metrics.

