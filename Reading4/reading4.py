import numpy as np

# Constants
h = 6.626e-27        # Planck constant in erg*s
c = 3e10             # speed of light in cm/s
lambda_v = 5500e-8   # V-band effective wavelength in cm
delta_lambda = 880   # V-band width in Angstroms

# Inputs
m_V = 24                  # Star magnitude
D = 4.0                   # Telescope diameter in meters
eta = 0.3                 # Camera efficiency
pixel_scale = 0.3         # arcsec/pixel
sigma_RN = 10             # electrons per pixel
dark_current = 0          # electrons/pixel/sec
theta_FWHM = 1.0          # arcsec
SNR_goal = 10

# 0th-magnitude flux in erg/cm^2/s/A
F0 = 3.6e-9

# Energy per photon
E_photon = h * c / lambda_v

# Photon flux for 0-mag star
F_photon_0 = F0 * delta_lambda / E_photon  # photons/cm^2/s

# Photon flux for the star
F_photon_star = F_photon_0 * 10**(-0.4 * m_V)

# Telescope area in cm^2
A_tel = np.pi * (D*100/2)**2

# Photon rate at detector
R_photon = F_photon_star * A_tel * eta
print(f"Photon rate at detector: {R_photon:.2f} photons/s")

# Aperture fraction (Gaussian PSF, 1 FWHM radius)
aperture_fraction = 0.76
R_ap = R_photon * aperture_fraction

# Number of pixels in aperture
r_pix = theta_FWHM / (2 * pixel_scale)  # radius in pixels
N_pix = int(np.pi * r_pix**2)

# Total read noise in aperture
sigma_RN_tot = np.sqrt(N_pix) * sigma_RN

# Background negligible
background = 0

# Compute integration time for desired SNR
# Solve SNR = R_ap * t / sqrt(R_ap*t + N_pix*sigma_RN^2)
a = R_ap
b = N_pix * sigma_RN**2
# Quadratic: (R_ap*t)^2 / (R_ap*t + b) = SNR^2 => R_ap^2 * t^2 - SNR^2 * R_ap * t - SNR^2 * b = 0
SNR = SNR_goal
coeff_a = a**2
coeff_b = -SNR**2 * a
coeff_c = -SNR**2 * b
t_sol = np.roots([coeff_a, coeff_b, coeff_c])
t_integration = np.max(t_sol)
print(f"Required integration time for S/N={SNR_goal}: {t_integration:.2f} s")

# Check if measurement is background-limited
if background * N_pix > sigma_RN_tot**2:
    print("Measurement is background-limited.")
else:
    print("Measurement is read-noise limited.")