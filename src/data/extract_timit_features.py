# Feature explanations:
#
# mean_spect, std_spect: Mean and standard deviation of the magnitude spectrogram (STFT). Capture overall energy and variability in frequency content.
# mean_spe, std_spe: Mean and standard deviation of the spectral centroid. The centroid indicates the "center of mass" of the spectrum, related to perceived brightness of the sound.
# zero_cross: Zero Crossing Rate. The rate at which the signal changes sign. Useful for distinguishing between voiced/unvoiced sounds and noise.
# rms_energ: Root Mean Square Energy. Measures the signal's power, related to loudness.
# mean_pitch, min_pitch, max_pitch, std_pitch: Statistics of the fundamental frequency (pitch). Pitch is important for characterizing speaker identity and gender.
# spectral_s, spectral_l: Mean spectral centroid and mean spectral bandwidth. Bandwidth measures the spread of the spectrum, related to timbre.
# energy_en: Signal energy. Total energy in the signal, related to loudness.
# log_energ: Logarithm of energy. Compresses the dynamic range, making features more robust.
# mfcc_1_m ... mfcc_13_m: Mean of Mel-Frequency Cepstral Coefficients (MFCCs) 1-13. MFCCs
#  represent the short-term power spectrum of sound, modeled on human hearing. They are widely used in speech and speaker recognition because they capture timbral and phonetic information.
# mfcc_1_st ... mfcc_13_st: Standard deviation of MFCCs 1-13.
#  Captures variability in the cepstral features, which can be informative for distinguishing speakers or phonemes.
# speaker_id: The TIMIT speaker ID (e.g., FJSP0, MKLS0). Encodes speaker identity and gender.
# gender: Speaker gender, inferred from speaker ID (F=Female, M=Male).
# file_path: Path to the audio file for traceability.

import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import logging

# Feature columns as in the user's screenshot
FEATURE_COLUMNS = [
    'mean_spect', 'std_spect', 'mean_spe', 'std_spe', 'mean_spe', 'zero_cross', 'rms_energ',
    'mean_pitch', 'min_pitch', 'max_pitch', 'std_pitch', 'spectral_s', 'spectral_l',
    'energy_en', 'log_energ',
] + [
    f"mfcc_{i}_m" for i in range(1, 14)
] + [
    f"mfcc_{i}_st" for i in range(1, 14)
] + [
    'speaker_id', 'gender', 'file_path', 'label', 'split'
]

# Helper to extract gender from speaker ID (TIMIT convention: F=Female, M=Male)
def extract_gender_from_speaker_id(speaker_id):
    if speaker_id[0].upper() == 'F':
        return 'female'
    elif speaker_id[0].upper() == 'M':
        return 'male'
    else:
        return 'unknown'

def gender_to_label(gender):
    if gender == 'male':
        return 0
    elif gender == 'female':
        return 1
    else:
        return -1  # unknown

def extract_features_from_file(file_path):
    y, sr = librosa.load(file_path, sr=None)
    S = np.abs(librosa.stft(y))
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zero_cross = np.mean(librosa.feature.zero_crossing_rate(y))
    rms_energ = np.mean(librosa.feature.rms(y=y))
    energy_en = np.sum(y ** 2) / len(y)
    log_energ = np.log(energy_en + 1e-8)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[magnitudes > np.median(magnitudes)]
    if len(pitches) > 0:
        mean_pitch = np.mean(pitches)
        min_pitch = np.min(pitches)
        max_pitch = np.max(pitches)
        std_pitch = np.std(pitches)
    else:
        mean_pitch = min_pitch = max_pitch = std_pitch = 0.0
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_stds = np.std(mfccs, axis=1)
    # Compose feature vector
    features = [
        np.mean(S), np.std(S), np.mean(spectral_centroid), np.std(spectral_centroid),
        np.mean(spectral_bandwidth), zero_cross, rms_energ,
        mean_pitch, min_pitch, max_pitch, std_pitch,
        np.mean(spectral_centroid), np.mean(spectral_bandwidth),
        energy_en, log_energ
    ]
    features += list(mfcc_means)
    features += list(mfcc_stds)
    return features

def process_timit_directory(root_dir, output_csv):
    logger = logging.getLogger(__name__)
    data = []
    logger.info(f"Starting feature extraction from TIMIT directory: {root_dir}")
    for split in ['TRAIN', 'TEST']:
        split_dir = Path(root_dir) / split
        logger.info(f"Processing split: {split_dir}")
        for dr in split_dir.glob('DR*'):
            logger.info(f"Processing dialect region: {dr.name}")
            for speaker in dr.iterdir():
                if not speaker.is_dir():
                    continue
                speaker_id = speaker.name
                gender = extract_gender_from_speaker_id(speaker_id)
                label = gender_to_label(gender)
                logger.info(f"Processing speaker: {speaker_id}")
                for wav_file in speaker.glob('*.WAV'):
                    try:
                        features = extract_features_from_file(str(wav_file))
                        row = features + [speaker_id, gender, str(wav_file), label, split]
                        data.append(row)
                    except Exception as e:
                        logger.error(f"Error processing {wav_file}: {e}")
    df = pd.DataFrame(data, columns=FEATURE_COLUMNS)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved features to {output_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract features from TIMIT audio files.")
    parser.add_argument('--timit_root', type=str, default='C:/Users/Asael/PycharmProjects/wonderful_mission/data/raw/full_timit/data', help='Path to TIMIT data/TRAIN and data/TEST directory')
    parser.add_argument('--output_csv', type=str, default='C:/Users/Asael/PycharmProjects/wonderful_mission/data/processed/timit_features.csv', help='Path to output CSV file')
    args = parser.parse_args()
    process_timit_directory(args.timit_root, args.output_csv) 