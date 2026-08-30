# -*- coding: utf-8 -*-
import numpy as np

def fresnel_propagate(wave_field, distance_m, pixel_size, wavelength):
    """
    Propagate a complex wave field with a simplified FFT-based Fresnel model.

    The input field is transformed into the spatial-frequency domain, where a
    quadratic phase transfer function is applied. An inverse FFT then returns
    the propagated complex field at the detector plane.

    Parameters
    ----------
    wave_field : numpy.ndarray
        Square, two-dimensional complex field at the object plane.
    distance_m : float
        Propagation distance between object and detector in meters.
    pixel_size : float
        Sampling pitch of the simulated detector in meters.
    wavelength : float
        Illumination wavelength in meters.

    Returns
    -------
    numpy.ndarray
        Complex-valued wave field at the detector plane.
    """
    N = wave_field.shape[0]
    kx = np.fft.fftfreq(N, d=pixel_size)
    ky = np.fft.fftfreq(N, d=pixel_size)
    KX, KY = np.meshgrid(kx, ky)
    fr = (pixel_size ** 2) / (wavelength * distance_m)
    kernel = np.exp(-1j * np.pi * (KX**2 + KY**2) / fr)
    field_fft = np.fft.fft2(wave_field)
    propagated_fft = field_fft * kernel
    return np.fft.ifft2(propagated_fft)

def create_sphere_phantom(size, radius, phase_shift):
    """
    Create a circular phase-only phantom for synthetic fringe generation.

    The phase decreases linearly from ``phase_shift`` at the center to zero at
    the specified radius. Pixels outside the circle have zero phase, so their
    complex transmission is one.

    Parameters
    ----------
    size : int
        Width and height of the square phantom in pixels.
    radius : int or float
        Radius of the circular phase object in pixels.
    phase_shift : float
        Maximum phase delay at the center of the object in radians.

    Returns
    -------
    numpy.ndarray
        Complex transmission function of the phase object.
    """
    Y, X = np.ogrid[:size, :size]
    center = size // 2
    dist = np.sqrt((X - center)**2 + (Y - center)**2)
    phase = phase_shift * (1 - dist / radius)
    phase[dist > radius] = 0.0
    return np.exp(1j * phase)

def generate_dataset(config):
    """
    Generate synthetic holograms and their axial-distance regression targets.

    For each sample, the function randomly chooses a propagation distance and
    phantom geometry, propagates the exit wave to the detector, and records the
    normalized intensity. The target is the signed deviation from the nominal
    distance of 1.00 m, expressed in centimeters.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration containing image size, number of samples,
        and the minimum and maximum propagation distances.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Holograms with shape ``(num_samples, size, size)`` and float targets
        with shape ``(num_samples,)``.
    """
    N = config['image_size']
    num_samples = config['num_samples']
    d_min = config['distance_min']
    d_max = config['distance_max']
    pixel_size = 6.5e-6
    wavelength = 1.1e-10

    X_data = []
    y_data = []

    for _ in range(num_samples):
        distance = np.random.uniform(d_min, d_max)
        radius = np.random.randint(15, 30)
        phase_shift = np.random.uniform(1.5, 3.0)

        exit_wave = create_sphere_phantom(N, radius, phase_shift)
        detector_wave = fresnel_propagate(exit_wave, distance, pixel_size, wavelength)
        hologram = np.abs(detector_wave) ** 2

        # Scale each hologram independently to the interval [0, 1]. This keeps
        # absolute intensity differences from dominating the network input.
        hologram = hologram / np.max(hologram)

        X_data.append(hologram)
        # Express the target relative to the nominal 1.00 m distance. For
        # example, 1.02 m becomes +2 cm and 0.98 m becomes -2 cm.
        deviation_cm = (distance - 1.0) * 100
        y_data.append(deviation_cm)

    return np.array(X_data, dtype=np.float32), np.array(y_data, dtype=np.float32)
