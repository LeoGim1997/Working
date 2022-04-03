from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
from pandas import array
from PIL import Image
from im import normalize


def compute_DFT(x: np.array) -> np.array:
    """Compute the Discrete Fourrier transform
    of a N-size 1-D signal array. \n
    dft[0] correspond to the DC components.
    dft[1:N/2] corresponds to positive frequencies.
    dft[N/2+1:] correspond to negative frequencies.

    Args:
        x (np.array) : discrete signal of size N.
    Returns:
        dft (np.array) : dft of the signal with complex coefficients.
    """
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    X = np.dot(e, x)
    return X


def frequency_resolution(fs: float, fc: float, lenght_fft: int) -> np.array:
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


def shift_frequencies(fs: float, f_samples: np.array) -> np.array:
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


def compute_dft_row(image: np.array) -> np.array:
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
        dft_h[i, :] = compute_DFT(image[i, :])
    return dft_h


np.setdiff1d


def compute_dft_col(image: np.array) -> np.array:
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
        dft_h[:, i] = compute_DFT(image[:, i])
    return dft_h


def compute_dft_image(image: np.array) -> np.array:
    """
    Function to compute the 2-D DFT transform of the input
    image. \\
    Args:
        image (np.array)
    Returns:
        dft_final (np.array): 2-D DFT image (complex values).
    """
    dft = compute_dft_row(image)
    dft_final = compute_dft_col(dft)
    return dft_final


def compute_cosinus(freq: float, t_final: float, amp=1) -> np.array:
    """Generate array values of a cosine signal observed between
    0 and t_final.
    The output signal will be of the form:\\
    amp*cos(2*pi*freq*t). \\
    Args:
        freq (float): desired frequency.
        t_final (float): final time of the observation.
        amp (float): Amplitude of the signal.
    Returns:
        signal (np.array): cosinus array.
    """
    step = 1. / float(30 * freq)
    t = np.arange(0, t_final, step)
    return t, amp * np.cos(2 * np.pi * freq * t)


def shift_frequency_row(image: np.array) -> np.array:
    N, M = np.shape(image)
    image_shifted = np.copy(image)
    if N % 2 == 0:
        image_shifted[0:int(N / 2), :] = image[int(N / 2):, :]
        image_shifted[int(N / 2) + 1:, :] = image[0:int(N / 2), :]
    else:
        image_shifted[:int(N / 2) + 1, :] = image[int(N / 2):]
        image_shifted[int(N / 2), :] = image[0, :]
        image_shifted[int(N / 2) + 1:, :] = image[0:int(N / 2), :]
    return image_shifted


def shift_frequency_col(image: np.array) -> np.array:
    N, M = np.shape(image)
    image_shifted = np.copy(image)
    if M % 2 == 0:
        image_shifted[:, 0:int(M / 2)] = image[:, int(M / 2 + 1):]
        image_shifted[:, int(M / 2) + 1:] = image[:, 0:int(M / 2)]
    else:
        image_shifted[:, :int(M / 2) + 1] = image[:, int(M / 2):]
        image_shifted[:, int(M / 2)] = image[:, 0]
        image_shifted[:, int(M / 2) + 1:] = image[:, 0:int(M / 2)]
    return image_shifted


def image_fftshift(dft_image: np.array) -> np.array:
    """Performs the fft_shift of the image to have
        Args :
            dft_image (np.array) : 2-D DFT of an image
    """
    img = shift_frequency_col(shift_frequency_row(dft_image))
    return img


def compute_DFT_inv(dft_image: np.array) -> np.array:
    N, M = np.shape(dft_image)
    norm = 1/float(N*M)
    return np.abs(norm*compute_dft_image(dft_image))


def plot_dft_image(img: np.array) -> None:
    """Functions for plotting after 2D-DFT transform
    For a better visualisation of the results,
    the final image frequencies are shiffted and the module
    is plot in log scale.
    Args:
        img (np.array) : input image
    """
    tf_image = compute_dft_image(img)
    a = compute_DFT_inv(img)
    tf_image = image_fftshift(np.abs(tf_image))
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(a, cmap='gray')
    plt.title('TF of the image')
    plt.show()


img = imread('/Users/leogimenez/Desktop/git_depo_local/Working/image/image_folder/Lena.jpeg')
plot_dft_image(img)
