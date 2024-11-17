# SAR Image Colorization with UNet2DModel


## 📖 Overview

This repository contains the implementation of a **Synthetic Aperture Radar (SAR) image colorization model** using a **UNet2DModel**. The model transforms grayscale SAR images into their optical (colorized) counterparts, enhancing the interpretability of SAR data for various applications such as remote sensing and environmental monitoring.

<p align="center">
  ![Screenshot 2024-11-17 193920](https://github.com/user-attachments/assets/febd39bc-a22f-4be3-8630-1b75294579d0)

</p>

---

## Features

- **Encoder-Decoder Architecture**: Built using the flexible UNet2DModel.
- **Attention Mechanisms**: Incorporated to improve the quality of colorization.
- **Custom Dataset Support**: Compatible with paired SAR and optical image datasets.
- **High Performance**: Optimized for efficient GPU usage and minimal memory consumption.
- **Training Logs**: Includes visualizations for loss, PSNR, and SSIM metrics.

