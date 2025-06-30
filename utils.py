import cv2
import matplotlib.pyplot as plt
from numba import njit, prange, int32, float32
import numpy as np
from scipy.ndimage import sobel, gaussian_filter, uniform_filter
from scipy.signal import fftconvolve
from scipy.optimize import minimize_scalar
import pandas as pd

# STEP 1: 
def convert_to_LMS(img, wavelength, L_interpolated, M_interpolated, S_interpolated):
    min_wavelength, max_wavelength = 400, 750
    visible_bands = np.where((wavelength >= min_wavelength) & (wavelength <= max_wavelength))[0]

    L_interpolated = L_interpolated[visible_bands]
    M_interpolated = M_interpolated[visible_bands]
    S_interpolated = S_interpolated[visible_bands]

    response = np.zeros((img.shape[0], img.shape[1], 3))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = img[i, j, :].squeeze()[visible_bands]
            response[i, j, 0] = np.sum(pixel * L_interpolated)
            response[i, j, 1] = np.sum(pixel * M_interpolated)
            response[i, j, 2] = np.sum(pixel * S_interpolated)
    return response 

# STEP 2: 
def normalize_LMS(LMS_image, target_max=1000):
    LMS_norm = LMS_image.astype(np.float32)
    max_val = np.percentile(LMS_norm, 99)
    max_val = max(max_val, 1e-6)
    LMS_norm = LMS_norm / max_val * target_max

    return LMS_norm

def calibration(LMS_image, target=1000):
    luminance_proxy = 0.68990272 * LMS_image[:, :, 0] + 0.34832189 * LMS_image[:, :, 1] 
    
    luminance_proxy = np.maximum(luminance_proxy, 1e-3)

    luminance_min = np.percentile(luminance_proxy, 1)
    luminance_max = np.percentile(luminance_proxy, 99)

    luminance_norm = np.clip((luminance_proxy - luminance_min) / (luminance_max - luminance_min), 0, 1)
    luminance_target = 1 + luminance_norm * (target - 1)

    scale_factor = luminance_target / (luminance_proxy + 1e-8)

    LMS_calibrated = LMS_image * scale_factor[:, :, np.newaxis]


    return LMS_calibrated


# gamma = 1.11
def calibrate_luminance_gamma_physical_multigray(L_proxy, gray_coords, gray_reflectances,
                                                  gamma_scene=1.0, L_max_scene=1000.0,
                                                  gamma_bounds=(0.5, 3.0)):
    
    L_targets = (np.array(gray_reflectances) ** gamma_scene) * L_max_scene

    L_measured = np.array([L_proxy[y, x] for (x, y) in gray_coords])

    def error_func(gamma):
        L_corrected = L_measured ** gamma
        rmse = np.sqrt(np.mean((L_corrected - L_targets) ** 2))
        return rmse

    res = minimize_scalar(error_func, bounds=gamma_bounds, method='bounded')
    best_gamma = res.x

    corrected_L = L_proxy ** best_gamma

    print(f"[Gamma Multi-Patch] Gamma stimato: {best_gamma:.4f} (RMSE: {res.fun:.2f})")
    print(f"[Target L]: {L_targets}")
    print(f"[L misurate]: {L_measured}")
    print(f"[L corrette]: {(L_measured ** best_gamma)}")

    return corrected_L, best_gamma


# STEP 3:
def glare_psf(theta, A, p):
    term1 = (1 - 0.08 * (A / 70)**4)

    t1 = 9.2e6 / ((1 + (theta / 0.0046)**2)**1.5)
    t2 = 1.5e5 / ((1 + (theta / 0.045)**2)**1.5)
    
    term2 = (1 + 1.6 * (A / 70)**4)
    t3 = (400 / (1 + (theta / 0.1)**2)) + 3e-8 * theta**2
    
    t4 = 1300 / ((1 + (theta / 0.1)**2)**1.5)
    t5 = 0.8 / ((1 + (theta / 0.1)**2)**0.5)
    
    glare = term1 * (t1 + t2) + term2 * t3 + p * (t4 + t5) + 2.5e-3 * p
    return glare

def apply_glare_LMS(LMS_image, psf):
    glared = np.zeros_like(LMS_image)
    for i in range(3):
        glared[:, :, i] = fftconvolve(LMS_image[:, :, i], psf, mode='same')
    return glared


# STEP 4:
@njit(parallel=True)
def rsr_single_channel(padded_log, h, w, dxs, dys, K, N, radius):
    output = np.zeros((h, w), dtype=np.float32)

    for y in prange(h):
        for x in range(w):
            cx, cy = x + radius, y + radius
            log_val = padded_log[cy, cx] 
            acc = 0.0

            for k in range(K):
                max_log = -1e6
                for n in range(N):
                    dx = dxs[k, n]
                    dy = dys[k, n]
                    sx = cx + dx
                    sy = cy + dy
             
                    if 0 <= sx < padded_log.shape[1] and 0 <= sy < padded_log.shape[0]:
                        neighbor_log = padded_log[sy, sx]
                        if neighbor_log > max_log:
                            max_log = neighbor_log

                acc += log_val - max_log
            
            output[y, x] = acc / K

    return output

# RSR mono-scale, single channels
def apply_rsr_LMS(LMS_image, radius=15, K=10, N=10, epsilon=1e-3):
    h, w, _ = LMS_image.shape
    LMS_rsr = np.zeros_like(LMS_image, dtype=np.float32)

    angles = np.random.rand(K, N) * 2 * np.pi
    radii = np.random.rand(K, N) * radius #  [0, radius]
    dxs = (radii * np.cos(angles)).astype(np.int32)
    dys = (radii * np.sin(angles)).astype(np.int32)


    dxs_nb = np.ascontiguousarray(dxs)
    dys_nb = np.ascontiguousarray(dys)

    for i in range(3):  # L, M, S
        channel = LMS_image[:, :, i]
        padded = np.pad(channel, radius, mode='reflect')
        padded_log = np.log(padded + epsilon)

        LMS_rsr[:, :, i] = rsr_single_channel(
            padded_log.astype(np.float32), h, w,
            dxs_nb, dys_nb, K, N, radius
        )

    return LMS_rsr

# multiscale RSR 
def apply_msrsr_LMS(LMS_image, radii_list, K=10, N=10, epsilon=1e-3):
    LMS_rsr_total = np.zeros_like(LMS_image, dtype=np.float32)

    for radius in radii_list:
        LMS_rsr = apply_rsr_LMS(
            LMS_image, radius=radius, K=K, N=N, epsilon=epsilon
        )
        LMS_rsr_total += LMS_rsr  

    LMS_rsr_avg = LMS_rsr_total / len(radii_list)
    return LMS_rsr_avg



# FASE 5:  
# 1^ MAtrix LMS → XYZ ( di Bradford: https://en.wikipedia.org/wiki/LMS_color_space)
M_XYZ_to_LMS = np.array([
    [0.4002, 0.7075, -0.0808],
    [-0.2263, 1.1653, 0.0457],
    [0.0000, 0.0000, 0.9182]
])
M_LMS_to_XYZ = np.linalg.inv(M_XYZ_to_LMS)

# 2^ Matrix XYZ -> RGB (http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html)
M_XYZ_to_RGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [0.0556434, -0.2040259,  1.0572252]
])

# 3^ Matrix LMS -> RGB
LMS_sRGB = M_XYZ_to_RGB @ M_LMS_to_XYZ 

def printMatrix():
    print("M_LMS_to_XYZ: ",M_LMS_to_XYZ)
    print("M_XYZ_to_LMS: ",M_XYZ_to_LMS)
    print("M_XYZ_to_RGB: ",M_XYZ_to_RGB)
    print("LMS_sRGB: ",LMS_sRGB)
    
# gamma https://it.wikipedia.org/wiki/Spazio_colore_sRGB 
def convert_LMS_to_RGB(LMS_image):
    """Converte immagine LMS in RGB standard"""
    rgb_image = np.dot(LMS_image, LMS_sRGB.T)
    rgb_image = np.clip(rgb_image, 0, None)
    
    for i in range(3):
        min_val, max_val = rgb_image[..., i].min(), rgb_image[..., i].max()
        if max_val - min_val > 0:
            rgb_image[..., i] = (rgb_image[..., i] - min_val) / (max_val - min_val)
 
    gamma_map = rgb_image > 0.0031308
    rgb_image[gamma_map] = 1.055 * np.power(rgb_image[gamma_map], 1.0 / 2.4) - 0.055
    rgb_image[~gamma_map] = 12.92 * rgb_image[~gamma_map]
    
    return (np.clip(rgb_image, 0, 1) * 255).astype(np.uint8)

# === STEP 6 ===
def compute_all_contrast_metrics(L_proxy, L_proxy_glare, L_proxy_rsr, kernel_size=11, sigma_list=[1, 2, 4, 8], edginess_n=1, sigma_b=2.0, sigma_l=8.0):

    def local_rms_contrast(img):
        img = img.astype(np.float32)
        mean = cv2.blur(img, (kernel_size, kernel_size))
        mean_sq = cv2.blur(img**2, (kernel_size, kernel_size))
        variance = np.maximum(mean_sq - mean**2, 0)
        return np.sqrt(variance)

    def gradient_magnitude_contrast(img):
        img = img.astype(np.float32)
        gx = sobel(img, axis=0)
        gy = sobel(img, axis=1)
        return np.sqrt(gx**2 + gy**2)

    def compute_dog_contrast_value(img):
        img = img.astype(np.float32)
        contrast_sum = np.zeros_like(img)
        for i in range(len(sigma_list) - 1):
            blur1 = gaussian_filter(img, sigma=sigma_list[i])
            blur2 = gaussian_filter(img, sigma=sigma_list[i + 1])
            contrast_sum += np.abs(blur1 - blur2)
        return np.mean(contrast_sum) / (len(sigma_list) - 1)

    def edginess_contrast(img):
        img = img.astype(np.float32)
        gx = sobel(img, axis=1)
        gy = sobel(img, axis=0)
        grad = np.sqrt(gx**2 + gy**2) + 1e-6
        phi = grad ** edginess_n
        num = uniform_filter(phi * img, size=kernel_size, mode='reflect')
        den = uniform_filter(phi, size=kernel_size, mode='reflect') + 1e-6
        E = num / den
        return np.abs(E - img) / (E + img + 1e-6)

    def ahumada_beard_contrast(img):
        img = img.astype(np.float32)
        B = gaussian_filter(img, sigma=sigma_b)
        L = gaussian_filter(B, sigma=sigma_l)
        contrast_map = np.abs(B - L) / (L + 1e-6)
        return np.mean(contrast_map)

    # Calcolo di tutti i contrasti per ciascuna immagine
    results = {
        "LMS": {
            "RMS":     np.mean(local_rms_contrast(L_proxy)),
            "Grad":    np.mean(gradient_magnitude_contrast(L_proxy)),
            "DoG":     compute_dog_contrast_value(L_proxy),
            "Edginess": np.mean(edginess_contrast(L_proxy)),
            "Ahumada": ahumada_beard_contrast(L_proxy),
        },
        "Glare": {
            "RMS":     np.mean(local_rms_contrast(L_proxy_glare)),
            "Grad":    np.mean(gradient_magnitude_contrast(L_proxy_glare)),
            "DoG":     compute_dog_contrast_value(L_proxy_glare),
            "Edginess": np.mean(edginess_contrast(L_proxy_glare)),
            "Ahumada": ahumada_beard_contrast(L_proxy_glare),
        },
        "RSR": {
            "RMS":     np.mean(local_rms_contrast(L_proxy_rsr)),
            "Grad":    np.mean(gradient_magnitude_contrast(L_proxy_rsr)),
            "DoG":     compute_dog_contrast_value(L_proxy_rsr),
            "Edginess": np.mean(edginess_contrast(L_proxy_rsr)),
            "Ahumada": ahumada_beard_contrast(L_proxy_rsr),
        }
        
    }

    
    df = pd.DataFrame(results).T.round(6)
    return df


# Plot
def plot_luminance_distributions(L_proxy, L_proxy_glare, L_proxy_rsr, bins=200):
    plt.figure(figsize=(12, 6))

    plt.hist(L_proxy.ravel(), bins=bins, alpha=0.6, label='Original', color='blue')
    plt.hist(L_proxy_glare.ravel(), bins=bins, alpha=0.6, label='Glare', color='orange')
    plt.hist(L_proxy_rsr.ravel(), bins=bins, alpha=0.6, label='RSR', color='green')

    plt.title('Proxy Luminance Distribution')
    plt.xlabel('Luminance value')
    plt.ylabel('Pixel number')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_luminance_distributions_single(L_proxy, bins=200):
    plt.figure(figsize=(12, 6))

    plt.hist(L_proxy.ravel(), bins=bins, alpha=0.6, label='Original', color='blue')

    plt.title('Proxy Luminance Distribution')
    plt.xlabel('Luminance value')
    plt.ylabel('Pixel number')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_single(LMS_image, Glare_image, RSR_image):    
    plt.figure(figsize=(8, 8))
    plt.imshow(LMS_image)
    plt.axis('off')
    plt.title('LMS')
    plt.savefig("LMS.png", bbox_inches='tight', pad_inches=0)
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.imshow(Glare_image)
    plt.axis('off')
    plt.title('Glare')
    plt.savefig("Glare.png", bbox_inches='tight', pad_inches=0)
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.imshow(RSR_image)
    plt.axis('off')
    plt.title('Multiscale RSR')
    plt.savefig("RGB_rsr.png", bbox_inches='tight', pad_inches=0)
    plt.show()




