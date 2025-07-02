# Computational Simulation of the Human Visual System.

This project implements a pipeline to simulate a computational model of the human visual system from hyperspectral images.  
A sequence of biologically plausible transformations was used to convert the image from raw spectral data to a visual representation in the sRGB color space.
![output](https://github.com/user-attachments/assets/ca953aa2-4f33-479d-b02a-72d6e907ab48)

---


## Dataset

The hyperspectral images used come from the official website of the [Color Imaging Lab of the University of Granada](https://colorimaginglab.ugr.es/pages/data), 
specifically from the **UGR Hyperspectral Image Database** section.

> To use the code correctly, download the images in `.tiff` format from that section and place them in the project folder.

---

## Implemented model.

The pipeline simulates some basic steps of human vision. The main steps are as follows:

### 1. Input: hyperspectral image
The hyperspectral image provides reflectance/spectral information in different bands. It is the starting point for simulation of the perceptual process.

### 2. Conversion to LMS space
Spectral data are converted to **LMS** space, which represents the responses of the three types of retinal cones (Long, Medium, Short).

### 3. Simulation of **Veiling Glare**
A model of **visual glare** (Veiling Glare) is applied, which is the glare caused by intraocular light scattering, reducing contrast and altering visual perception.

### 4. **Multiscale Retinex (MRSR)**
Application of the **Multiscale model of Random Spray Retinex**, to simulate the neural mechanisms of illumination and spatial compensation.

### 5. Final rendering in sRGB
The final result is converted to the **sRGB** color space, for proper display on standard monitors.

---

## Requirements
- Python 3.x
- NumPy
- OpenCV
- SciPy
- Matplotlib
- Tifffile
- Pandas
- Numba
- Cv2

