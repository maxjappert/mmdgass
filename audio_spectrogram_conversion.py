import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image

chunk_length = 5  # in seconds
n_fft = 2048
hop_length = 512


def audio_to_spectrogram(audio, name, save_to_file=False, dest_folder=None, from_file=False, sr=22050):
    """
    Converts real-space audio file to spectrogram using librosa.

    Args:
        audio (str/ndarray): Audio to be converted to spectrogram.
        name (str): Name of output file.
        save_to_file (bool): True if you want to save spectrogram to file.
        dest_folder (Path): Destination folder where spectrogram will be saved if save_to_file is True.
        from_file (bool): True if initial audio is loaded from file.
        sr (int): Sample rate in Hz.

    Returns:
        ndarray, ndarray: Magnitude spectrogram and corresponding phase.
    """
    if from_file:
        audio, _ = librosa.load(audio, sr=sr)

    # Convert to spectrogram
    S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    phase = np.angle(S)

    # Normalise the spectrogram to 0-255
    S_db_normalized = (S_db + 80) / 80 * 255
    S_db_normalized = S_db_normalized.astype(np.uint8)

    # Save the spectrogram to a PNG file
    if save_to_file and dest_folder:
        plt.imsave(dest_folder / f'{name}.png', S_db_normalized, cmap='gray')
        np.save(dest_folder / f'{name}_phase.npy', S_db_normalized)

    return S_db_normalized, phase


def spectrogram_to_audio(spectrogram, output_filename, phase=None, from_file=False, sr=22050):
    """
    Converts a spectrogram to audio.

    Args:
        spectrogram (str/ndarray): Spectrogram either as file path or numpy array.
        output_filename (str): File the audio should be saved to.
        phase (ndarray): Phase information of spectrogram.
        from_file (bool): True if spectrogram shall be loaded from a file.
        sr (int): Sample rate in Hz.

    Returns:
        ndarray: Audio signal in real space.
    """
    # Load the spectrogram image
    if from_file:
        img = Image.open(f'{spectrogram}').convert('L')
        if phase:
            phase = np.load(phase)
        S_db_imported = np.array(img)
    else:
        S_db_imported = spectrogram * 255 if spectrogram.max() <= 1 else spectrogram
        #print(S_db_imported)

    # Convert back to the original dB scale
    S_db_imported = S_db_imported.astype(np.float32) / 255 * 80 - 80

    # Convert back to amplitude
    S_imported = librosa.db_to_amplitude(S_db_imported)

    # Use Griffin-Lim algorithm to approximate the phase and reconstruct the audio
    if phase is None:
        audio_signal = librosa.core.griffinlim(S_imported)
    else:
        assert S_imported.shape == phase.shape, f'{S_imported.shape} mismatched {phase.shape}'
        audio_signal = librosa.istft(S_imported * np.exp(1j * phase))

    if output_filename is not None:
        # Write the output to a WAV file
        scipy.io.wavfile.write(f'wavs/{output_filename}.wav', sr, np.array(audio_signal * 32767, dtype=np.int16))

    return audio_signal * 32767
