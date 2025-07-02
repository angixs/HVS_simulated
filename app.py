import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import tifffile as tiff
import utils

# === STEP 1: Hyperspectral image retrieval -> Conversion to LMS ===
tiff_path = "img1.tiff"

hsi_data = tiff.imread(tiff_path).astype(np.float32) / 65536.0

if hsi_data.shape[0] == 61:
    hsi_data = np.transpose(hsi_data, (1, 2, 0))

wavelengths = np.arange(400, 1001, 10)

sensitivity_data = pd.read_csv("linss2_10e_5.csv")
sensitivity_data.columns = sensitivity_data.columns.str.strip()

wave_cmf = sensitivity_data.iloc[:, 0].values
L_cmf = sensitivity_data.iloc[:, 1].values
M_cmf = sensitivity_data.iloc[:, 2].values
S_cmf = np.nan_to_num(sensitivity_data.iloc[:, 3].values)

L_cmf /= np.trapezoid(L_cmf, wave_cmf)
M_cmf /= np.trapezoid(M_cmf, wave_cmf)
S_cmf /= np.trapezoid(S_cmf, wave_cmf)

L_interp = PchipInterpolator(wave_cmf, L_cmf)(wavelengths)
M_interp = PchipInterpolator(wave_cmf, M_cmf)(wavelengths)
S_interp = PchipInterpolator(wave_cmf, S_cmf)(wavelengths)

LMS_image = utils.convert_to_LMS(hsi_data, wavelengths, L_interp, M_interp, S_interp)

#=== STEP 2: Luminance calibration ===
LMS_normalized = utils.normalize_LMS(LMS_image, target_max=1000)
LMS_cal = utils.calibration(LMS_normalized, 1000)

L_proxy = 0.68990272 * LMS_cal[:, :, 0] + 0.34832189 * LMS_cal[:, :, 1]

best_gamma = 1.11

LMS_cal[:, :, 0] = LMS_cal[:, :, 0] ** best_gamma
LMS_cal[:, :, 1] = LMS_cal[:, :, 1] ** best_gamma
LMS_cal[:, :, 2] = LMS_cal[:, :, 2] ** best_gamma
L_proxy_corr = 0.68990272 * LMS_cal[:, :, 0] + 0.34832189 * LMS_cal[:, :, 1]

LMS_gamma = LMS_cal

RGB_image = utils.convert_LMS_to_RGB(LMS_gamma)

# === STEP 3: Application of Veiling Glare ===
a = 50  # age
p = 1   # pigmentation 
fov = 4 # angle
res = 1 # resolution

angle_range = np.arange(-fov/2, fov/2 + res, res)
X, Y = np.meshgrid(angle_range, angle_range)
theta = np.sqrt(X**2 + Y**2)

psf = utils.glare_psf(theta, a, p)
psf_compressed = np.log1p(psf)
psf_compressed /= psf_compressed.sum()

LMS_glared = utils.apply_glare_LMS(LMS_gamma, psf_compressed)
RGB_glared = utils.convert_LMS_to_RGB(LMS_glared)

# === STEP 4: RSR Calculation - Random Spray Retinex ===
radii = [ 30, 150, 300, 600]
# Case 1: img with subjects filmed far away. 
K = 5
N = 15
alpha = 0.5

# Case 2: img with subjects shot close together.
#K = 5
#N = 18
#alpha = 0.2


LMS_rsr = utils.apply_msrsr_LMS(LMS_glared, radii_list=radii, K=K, N=N)
LMS_fused = LMS_glared * np.exp(alpha * LMS_rsr)



# === STEP 5: LMS_rsr → RGB conversion ===
RGB_rsr = utils.convert_LMS_to_RGB(LMS_fused)

fig, axes = plt.subplots(1, 3, figsize=(24, 6))
axes[0].imshow(RGB_image)
axes[0].set_title('Original LMS')
axes[0].axis('off')

axes[1].imshow(RGB_glared)
axes[1].set_title('Post Glare')
axes[1].axis('off')

axes[2].imshow(RGB_rsr)
axes[2].set_title('MRSR')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# === STEP 6: Contrast assessment ===
L_proxy = 0.68990272 * LMS_cal[:, :, 0] + 0.34832189 * LMS_cal[:, :, 1]
L_proxy_glare = 0.68990272 * LMS_glared[:, :, 0] + 0.34832189 * LMS_glared[:, :, 1]
L_proxy_rsr = 0.68990272 * LMS_fused[:, :, 0] + 0.34832189 * LMS_fused[:, :, 1]
utils.compute_all_contrast_metrics(L_proxy, L_proxy_glare, L_proxy_rsr)