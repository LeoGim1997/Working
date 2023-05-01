import numpy as np
import matplotlib.pyplot as plt
from processing.im import rotating_image
from typing import Tuple


def compute_DFT(x: np.ndarray, inv: bool = False) -> np.ndarray:
    """Compute the Discrete Fourrier transform
    of a N-size 1-D signal array. \n
    dft[0] correspond to the DC components.
    dft[1:N/2] corresponds to positive frequencies.
    dft[N/2+1:] correspond to negative frequencies.

    Args:
        x (array_like) : discrete signal of size N.
        inv (Optional bool) : Is set to true, will invert the frequency vector init. Default to False.
    Returns:
        dft (ndarray) : dft of the signal with complex coefficients.
    """
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    if inv:
        e = np.exp(2j * np.pi * k * n / N)
    else:
        e = np.exp(-2j * np.pi * k * n / N)
    X = np.dot(e, x)
    return X


def frequency_resolution(fs: float, fc: float, lenght_fft: int) -> np.ndarray:
    """Create a frequency vector for plotting.
    When computing DFT the returned DFT vectors is indexed by integers.
    For interpreting, we need to generate a frequency resolution
    vectors which contains evenly spaced frequency values.\\
    Args:
        fs (float): sampling frequency used.
        fc (float): frequency of the carrier.
        lenght_fft (float): number of samples in DFT array.
    Returns:
    """
    step = (fs * fc) / lenght_fft
    return step * np.linspace(0, lenght_fft, num=lenght_fft)


def shift_frequencies(fs: float, f_samples: np.ndarray) -> np.ndarray:
    """Shift the frequencies values for centering the spectrum around zero.
    before shifting the N/2 indexed values correspond to the
    Nyquist index.\\
    After the shifting, frequencies values go from [-f,f].\\
    Args:
        fs (float): sampling frequency.
        f_sampling (float): frequency array.
    Returns:
        np.array: shifted frequencies array.
    """
    N = len(f_samples)
    d_fs = fs / N
    if N % 2 == 0:
        f_shift = np.arange(int(N / 2), int(N / 2), 1)
        f_shift = d_fs * f_shift
    f_shift = np.arange(-1 * int(N / 2), int((N + 1) / 2), 1)
    f_shift = d_fs * f_shift
    return f_shift


def compute_dft_row(image: np.ndarray, inv=False) -> np.ndarray:
    """
    Apply DFT on every row of the input image.
    Will returns the complex horizontal fourrier coefficients
    image. \\
    Args:
        image (np.array)
    Returns:
        dft_h (np.array)
    """
    N, M = np.shape(image)
    dft_h = np.zeros((N, M), dtype=complex)
    for i in range(N):
        dft_h[i, :] = compute_DFT(image[i, :], inv)
    return dft_h


def compute_dft_col(image: np.ndarray, inv=False) -> np.ndarray:
    """
    Apply DFT on every columns of the input image.
    Will returns the complex horizontal fourrier coefficients
    image. \\
    Args:
        image (np.array)
    Returns:
        dft_h (np.array)
    """
    N, M = np.shape(image)
    dft_h = np.zeros((N, M), dtype=complex)
    for i in range(M):
        dft_h[:, i] = compute_DFT(image[:, i], inv=False)
    return dft_h


def compute_dft_image(image: np.ndarray, inv=False) -> np.ndarray:
    """
    Function to compute the 2-D DFT transform of the input
    image. \
    Args:
        image (np.array)
    Returns:
        dft_final (np.array): 2-D DFT image (complex values).
    """
    dft = compute_dft_row(image, inv)
    dft_final = compute_dft_col(dft, inv)
    return dft_final


def compute_cosinus(
    freq: float, t_final: float, amp=1, nsamples=None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate array values of a cosine signal observed between
    0 and t_final.
    The output signal will be of the form:\\
    amp*cos(2*pi*freq*t). \\
    Args:
        freq (float): desired frequency.
        t_final (float): final time of the observation.
        amp (float): Amplitude of the signal.
        step (Optional-int): When not None, specify the number of samples
            between 0 and t_final.
    Returns:
        t (np.array): the time array.
        signal (np.array): cosinus array.
    """
    if nsamples:
        t = np.linspace(0, t_final, nsamples)
    else:
        step = 1.0 / float(10 * freq)
        t = np.arange(0, t_final, step)
    return t, amp * np.cos(2 * np.pi * freq * t)


def shift_frequency_row(image: np.ndarray) -> np.ndarray:
    N, M = np.shape(image)
    image_shifted = np.copy(image)
    if N % 2 == 0:
        image_shifted[0 : int(N / 2), :] = image[int(N / 2) :, :]
        image_shifted[int(N / 2) + 1 :, :] = image[0 : int(N / 2), :]
    else:
        image_shifted[: int(N / 2) + 1, :] = image[int(N / 2) :]
        image_shifted[int(N / 2), :] = image[0, :]
        image_shifted[int(N / 2) + 1 :, :] = image[0 : int(N / 2), :]
    return image_shifted


def shift_frequency_col(image: np.ndarray) -> np.ndarray:
    N, M = np.shape(image)
    image_shifted = np.copy(image)
    if M % 2 == 0:
        image_shifted[:, 0 : int(M / 2)] = image[:, int(M / 2 + 1) :]
        image_shifted[:, int(M / 2) + 1 :] = image[:, 0 : int(M / 2)]
    else:
        image_shifted[:, : int(M / 2) + 1] = image[:, int(M / 2) :]
        image_shifted[:, int(M / 2)] = image[:, 0]
        image_shifted[:, int(M / 2) + 1 :] = image[:, 0 : int(M / 2)]
    return image_shifted


def image_fftshift(dft_image: np.ndarray) -> np.ndarray:
    """Performs the fft_shift of the image to get the corresponding
    centered spectrum.
        Args :
            dft_image (np.array) : 2-D DFT of an image
    """
    img = shift_frequency_col(shift_frequency_row(dft_image))
    return img


def compute_DFT_inv(dft_image: np.ndarray) -> np.ndarray:
    N, M = np.shape(dft_image)
    norm = 1 / float(N * M)
    return rotating_image(np.real(norm * compute_dft_image(dft_image, inv=True)))


def plot_dft_image(img: np.ndarray, superpose: bool = False) -> None:
    """Functions for plotting after 2D-DFT transform
    For a better visualisation of the results,
    the final image frequencies are shiffted and the module
    is plot in log scale.
    Args:
        img (np.array) : input image
        superpose (bool) : If set to True, will returns the superposition of module and
                    phase of the DFT image. Default to False.

    """
    tf_image = compute_dft_image(img)
    module = np.abs(image_fftshift(tf_image))
    phase = np.angle(image_fftshift(tf_image))
    plt.figure()
    if superpose:
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap="gray")
        plt.title("Original Image")
        plt.subplot(1, 2, 2)
        plt.imshow(np.log(module) + phase, cmap="gray")
        plt.title("FT Transform in log scale of Image")
    else:
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap="gray")
        plt.title("Original Image")
        plt.subplot(1, 3, 2)
        plt.imshow(np.log(module), cmap="gray")
        plt.title("FT Transform in log scale of Image")
        plt.subplot(1, 3, 3)
        plt.imshow(phase, cmap="gray")
        plt.title("Phase of the FT (rad)")
    plt.show()
