import librosa
import numpy as np

from bs4 import BeautifulSoup
from os.path import join as ospj

from .globals import BASEPATH
from .globals import SAMPLING_RATE


def calculate_melspectrogram(y: np.ndarray, sr: int) -> np.ndarray:
    spec = np.abs(librosa.feature.melspectrogram(y=y, sr=SAMPLING_RATE))
    spec = librosa.amplitude_to_db(spec, ref=np.max)
    
    return spec


def create_annotation_matrix(events: list, num_frames: int) -> np.ndarray:
    instrument2index = {'HH': 0, 'SD': 1, 'KD': 2}
    annotations = np.zeros((3, num_frames), dtype=np.float32)
    
    for event in events:
        onset = float(event.onsetsec.string)
        instrument = event.instrument.string
        
        index = instrument2index[instrument]
        onset = librosa.time_to_frames(onset, sr=SAMPLING_RATE)
        annotations[index, onset] = 1.0
    
    return annotations


def create_feature_and_annotation(songname: str) -> tuple:
    audiofile = ospj(BASEPATH, f'audio/{songname}.wav')
    annotationfile = ospj(BASEPATH, f'annotation/{songname}.xml')
    
    with open(annotationfile, 'r') as fp:
        soup = BeautifulSoup(fp, 'lxml')
        events = soup.find_all('event')
    
    wave, sr = librosa.load(audiofile, sr=SAMPLING_RATE)
    spec = calculate_melspectrogram(wave, sr=sr)
    annotation = create_annotation_matrix(events, spec.shape[1])
    
    return spec, annotation


def chunkify(songname, window_size=256, hop_length=64):
    spec, annotation = create_feature_and_annotation(songname)
    num_frames = spec.shape[1]
    
    for i in range(0, num_frames - window_size + 1, hop_length):
        spec_chunk = spec[:, i:i+window_size]
        annotation_chunk = annotation[:, i:i+window_size]
        
        yield spec_chunk, annotation_chunk