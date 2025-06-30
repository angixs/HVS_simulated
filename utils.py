import cv2
import matplotlib.pyplot as plt
from numba import njit, prange, int32, float32
import numpy as np
from scipy.ndimage import sobel, gaussian_filter, uniform_filter
from scipy.signal import fftconvolve
from scipy.optimize import minimize_scalar
import pandas as pd
# FASE 1: Recupero immagine Iperspettrale -> Conversione in LMS 
def convert_to_LMS(img, wavelength, L_interpolated, M_interpolated, S_interpolated):
    # definisce l'intervallo visibile
    min_wavelength, max_wavelength = 400, 750
    visible_bands = np.where((wavelength >= min_wavelength) & (wavelength <= max_wavelength))[0]

    #  filtra le onde in input sulla base dell'intervallo visibile
    L_interpolated = L_interpolated[visible_bands]
    M_interpolated = M_interpolated[visible_bands]
    S_interpolated = S_interpolated[visible_bands]

    # inizializzazione immagine per l'output
    response = np.zeros((img.shape[0], img.shape[1], 3))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # estrae lo spettro visibile e calcola LMS come prodotto scalare tra 
            # il vettore spettrale del pixel e la sensibilità del cono
            # permette di proiettare il contenuto spettrale sulla base delle risposte dei coni
            pixel = img[i, j, :].squeeze()[visible_bands]
            response[i, j, 0] = np.sum(pixel * L_interpolated)
            response[i, j, 1] = np.sum(pixel * M_interpolated)
            response[i, j, 2] = np.sum(pixel * S_interpolated)
    return response 

# FASE 2: Calibrazione luminanza
def normalize_LMS(LMS_image, target_max=1000):
    # normalizzo l'imamgine per evitare problemi numeri e avere una scala coerente
    # prendendo in input l'immagine LMS, andando a normalizzare in modo percentile
    LMS_norm = LMS_image.astype(np.float32)
    max_val = np.percentile(LMS_norm, 99)
    max_val = max(max_val, 1e-6)
    LMS_norm = LMS_norm / max_val * target_max

    return LMS_norm

def calibration(LMS_image, target=1000):
    # calcola un approsimazione di luminanza
    # luminance_proxy = 0.6 * L_channel + 0.4 * M_channel
    # secondo CIE standard 2006 - Stockman & Sharpe
    luminance_proxy = 0.68990272 * LMS_image[:, :, 0] + 0.34832189 * LMS_image[:, :, 1] 
    
    # filtro di sicurezza per evitare valori alti
    luminance_proxy = np.maximum(luminance_proxy, 1e-3)

    # stretch tra percentili per evitare outlier estremi
    luminance_min = np.percentile(luminance_proxy, 1)
    luminance_max = np.percentile(luminance_proxy, 99)

    # normalizzazione [0, 1]
    luminance_norm = np.clip((luminance_proxy - luminance_min) / (luminance_max - luminance_min), 0, 1)
    luminance_target = 1 + luminance_norm * (target - 1)

    # fattore di scala (con stabilizzatore)
    scale_factor = luminance_target / (luminance_proxy + 1e-8)

    # applica la scala all’intera LMS
    LMS_calibrated = LMS_image * scale_factor[:, :, np.newaxis]

    # luminance_calibrata = 0.68990272 * LMS_calibrated[:, :, 0] +0.34832189  * LMS_calibrated[:, :, 1]
    # luminance_calibrata = np.clip(luminance_calibrata, 1, target)

    return LMS_calibrated


# trovare la gamma, l'ho usata per definire gamma con immagine con color cheker
# gamma = 1.11
def calibrate_luminance_gamma_physical_multigray(L_proxy, gray_coords, gray_reflectances,
                                                  gamma_scene=1.0, L_max_scene=1000.0,
                                                  gamma_bounds=(0.5, 3.0)):
    
    # luminanze target teoriche (da riflettanze note)
    L_targets = (np.array(gray_reflectances) ** gamma_scene) * L_max_scene

    # luminanze misurate nei pixel corrispondenti
    L_measured = np.array([L_proxy[y, x] for (x, y) in gray_coords])

    # obiettivo: RMSE tra luminanze corrette e target
    def error_func(gamma):
        L_corrected = L_measured ** gamma
        rmse = np.sqrt(np.mean((L_corrected - L_targets) ** 2))
        return rmse

    # ottimizzazione del gamma
    res = minimize_scalar(error_func, bounds=gamma_bounds, method='bounded')
    best_gamma = res.x

    # applica correzione gamma su tutta l'immagine
    corrected_L = L_proxy ** best_gamma

    print(f"[Gamma Multi-Patch] Gamma stimato: {best_gamma:.4f} (RMSE: {res.fun:.2f})")
    print(f"[Target L]: {L_targets}")
    print(f"[L misurate]: {L_measured}")
    print(f"[L corrette]: {(L_measured ** best_gamma)}")

    return corrected_L, best_gamma


# FASE 3: Applicazione del Veiling Glare
# funzione glare (formula 8 semplificata dalla letteratura)
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

# funzione per applicare glare a tutti i canali LMS, in modo separato
# ha in input la psf: kernel 2D che modella come la luce si "diffonde" o si sparge su un sensore 
def apply_glare_LMS(LMS_image, psf):
    glared = np.zeros_like(LMS_image)
    # cicla sui 3 canali e applica una convoluzione 2D tra il canale e il PSF con fftconvolve
    # fftconvolve -> convoluzione veloce tramite FFT
    # per spargere ongi punto luminoso secondo la forma della psf
    for i in range(3):
        glared[:, :, i] = fftconvolve(LMS_image[:, :, i], psf, mode='same')
    return glared


# FASE 4: Calcolo RSR - Random Spray Retinex 
# velocizzo con numba
@njit(parallel=True)
def rsr_single_channel(padded_log, h, w, dxs, dys, K, N, radius):
    output = np.zeros((h, w), dtype=np.float32)

    for y in prange(h):
        for x in range(w):
            cx, cy = x + radius, y + radius
            log_val = padded_log[cy, cx] # valore logaritmico del pixel centrale
            acc = 0.0

            for k in range(K):
                max_log = -1e6
                for n in range(N):
                    dx = dxs[k, n]
                    dy = dys[k, n]
                    sx = cx + dx
                    sy = cy + dy
                # trova il valore massimo nello spray
                    if 0 <= sx < padded_log.shape[1] and 0 <= sy < padded_log.shape[0]:
                        neighbor_log = padded_log[sy, sx]
                        if neighbor_log > max_log:
                            max_log = neighbor_log

                acc += log_val - max_log
            # media dei contrasti su K spray
            output[y, x] = acc / K

    return output

# applicazione RSR mono-scala, per singoli canali, restituisce l'immagine 
def apply_rsr_LMS(LMS_image, radius=15, K=10, N=10, epsilon=1e-3):
    h, w, _ = LMS_image.shape
    LMS_rsr = np.zeros_like(LMS_image, dtype=np.float32)

    angles = np.random.rand(K, N) * 2 * np.pi
    radii = np.random.rand(K, N) * radius # raggi casuali [0, radius]
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

# composizione multiscala del RSR con la media dei diversi raggi
def apply_msrsr_LMS(LMS_image, radii_list, K=10, N=10, epsilon=1e-3):
    LMS_rsr_total = np.zeros_like(LMS_image, dtype=np.float32)

    for radius in radii_list:
        LMS_rsr = apply_rsr_LMS(
            LMS_image, radius=radius, K=K, N=N, epsilon=epsilon
        )
        LMS_rsr_total += LMS_rsr  # somma i risultati

    # media delle scale
    LMS_rsr_avg = LMS_rsr_total / len(radii_list)
    return LMS_rsr_avg



# FASE 5: Conversione LMS_rsr → RGB 
# 1^ Matrice LMS → XYZ (inversa di Bradford: https://en.wikipedia.org/wiki/LMS_color_space)
M_XYZ_to_LMS = np.array([
    [0.4002, 0.7075, -0.0808],
    [-0.2263, 1.1653, 0.0457],
    [0.0000, 0.0000, 0.9182]
])
M_LMS_to_XYZ = np.linalg.inv(M_XYZ_to_LMS)

# 2^ Matrice XYZ -> RGB (http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html)
M_XYZ_to_RGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [0.0556434, -0.2040259,  1.0572252]
])

# 3^ Matrice LMS -> RGB
LMS_sRGB = M_XYZ_to_RGB @ M_LMS_to_XYZ 

def printMatrix():
    print("M_LMS_to_XYZ: ",M_LMS_to_XYZ)
    print("M_XYZ_to_LMS: ",M_XYZ_to_LMS)
    print("M_XYZ_to_RGB: ",M_XYZ_to_RGB)
    print("LMS_sRGB: ",LMS_sRGB)
    
# funzione di conversone finale
# gamma https://it.wikipedia.org/wiki/Spazio_colore_sRGB 
def convert_LMS_to_RGB(LMS_image):
    """Converte immagine LMS in RGB standard"""
    rgb_image = np.dot(LMS_image, LMS_sRGB.T)
    rgb_image = np.clip(rgb_image, 0, None)
    
    # normalizza canale per canale
    for i in range(3):
        min_val, max_val = rgb_image[..., i].min(), rgb_image[..., i].max()
        if max_val - min_val > 0:
            rgb_image[..., i] = (rgb_image[..., i] - min_val) / (max_val - min_val)

    # correzione gamma (approssimata a sRGB: )
    gamma_map = rgb_image > 0.0031308
    rgb_image[gamma_map] = 1.055 * np.power(rgb_image[gamma_map], 1.0 / 2.4) - 0.055
    rgb_image[~gamma_map] = 12.92 * rgb_image[~gamma_map]
    
    return (np.clip(rgb_image, 0, 1) * 255).astype(np.uint8)

# === FASE 6. Valutazione del contrasto ===
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

def compute_metrics(luminance_map):
    img = luminance_map.astype(np.float32)
    img_norm = np.clip(img / np.max(img), 0, 1)

    rms = np.std(img)
    michelson = (np.max(img) - np.min(img)) / (np.max(img) + np.min(img) + 1e-6)
    dyn_range = np.percentile(img, 99) - np.percentile(img, 1)

    return {
        "RMS": rms,
        "Michelson": michelson,
        "Dynamic Range": dyn_range,
    }



# Sezione dei Plot
def plot_gamma_curve(gamma):
    x = np.linspace(0, 1, 1000)
    y = x ** gamma
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, label=f'Gamma = {gamma:.2f}')
    plt.title('Curva di luminanza applicata')
    plt.xlabel('Luminanza normale (x)')
    plt.ylabel('Luminanza corretta (x^γ)')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_luminance_distributions(L_proxy, L_proxy_glare, L_proxy_rsr, bins=200):
    plt.figure(figsize=(12, 6))

    plt.hist(L_proxy.ravel(), bins=bins, alpha=0.6, label='Originale', color='blue')
    plt.hist(L_proxy_glare.ravel(), bins=bins, alpha=0.6, label='Glare', color='orange')
    plt.hist(L_proxy_rsr.ravel(), bins=bins, alpha=0.6, label='RSR', color='green')

    plt.title('Distribuzione della Luminanza Proxy')
    plt.xlabel('Valore di luminanza')
    plt.ylabel('Numero di pixel')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_luminance_distributions_single(L_proxy, bins=200):
    plt.figure(figsize=(12, 6))

    plt.hist(L_proxy.ravel(), bins=bins, alpha=0.6, label='Originale', color='blue')

    plt.title('Distribuzione della Luminanza Proxy')
    plt.xlabel('Valore di luminanza')
    plt.ylabel('Numero di pixel')
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
    plt.title('Multiscala RSR')
    plt.savefig("RGB_rsr.png", bbox_inches='tight', pad_inches=0)
    plt.show()




