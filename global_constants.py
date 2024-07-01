# Preprocessing
base_music_format = '.mp3'
from_patterns = ['*.flac', '*.wav', '*.mp3', '*.m4a']
encode_params = '-vn -ar 44100 -ac 2 -b:a 256k'
disk_sample_rate = 44100
sample_rate = 44100
MFCC = 'mfcc'
STFT = 'stft'
RAW = 'raw'


# Train
train_split = 0.5
val_split = 0.25
test_split = 1 - (train_split + val_split)
