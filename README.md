# MLDM Project 2023

## Overview

Welcome to the MLDM Project repository. This project is part of the Master's program in Machine Learning and Data Mining at Jean Monnet University. Our team is focused on exploring the utilization of the Sparse Identification of Nonlinear Dynamics (SINDy) algorithm to find partial differential equations (PDEs) from sets of unregistered microscopy images of surfaces exhibiting self-arranging patterns upon stimulation.

## Project Description

The primary challenge addressed in this project is the identification of governing PDEs from unregistered electron microscopy images. These images exhibit patterns similar to those produced by the 2D Swift-Hohenberg equation. Our approach involves two main steps:

1. **Image Registration:** We apply various image registration methods to align the unregistered electron microscopy images. This is a crucial preprocessing step to ensure the accuracy of subsequent analyses.
2. **PDE Identification:** After registration, we use the SINDy algorithm and its variant, SINDyCP, to identify the governing PDEs from the processed images.

## Repository Contents

- `source/` - Contains all source code for the project.
  - Image registration methods.
  - PDE identification using SINDy and SINDyCP.
  - Preprocessing and visualization utilities.
  - Analysis scripts.
- `generations/` - Contains generated data from the original images as well as the results of the image registration process, and some product of Swift-Hohenberg solvers.
- Additional support files and documentation.

### Prerequisites

- See `requirements.txt` for a list of required Python packages.

## Authors and Acknowledgment

- Made by Ariel Guerra-Adames, Felipe Jaramillo Cortes, Bastian Schäfer and Franck Sirguey.
- Supervised by Marc Sebban and Remi Emonet.