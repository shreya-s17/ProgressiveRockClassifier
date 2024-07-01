

class MusicSample(object):
    def __init__(self, audio_seq, filename, mfcc, meta):
        self.audio_seq = audio_seq
        self.filename = filename
        self.mfcc = mfcc
        self.meta = meta
