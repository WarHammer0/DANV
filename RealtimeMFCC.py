import pyaudio
from python_speech_features import mfcc
import numpy as np
import torch


CHUNK = 10240
WIDTH = 2
CHANNELS = 1
RATE = 44100
FORMAT = pyaudio.paInt16

p = pyaudio.PyAudio()

stream = p.open(format=p.get_format_from_width(WIDTH),
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK)

print("* recording")

while True:
    data = stream.read(CHUNK)
    sig = np.fromstring(data, 'Int16');

    print(sig.shape)
    stream.write(data, CHUNK)
    mfcc_feat = zip(*mfcc(sig[:int(44100 * 0.2)],
                          RATE,
                          winlen=0.025,
                          winstep=0.0095,
                          numcep=12,
                          nfilt=26,
                          nfft=2048,
                          lowfreq=0,
                          highfreq=None,
                          preemph=0.97,
                          ceplifter=22,
                          appendEnergy=True))
    mfcc_feat = np.stack([np.array(i) for i in mfcc_feat])
    print(mfcc_feat.shape)

    cc = np.expand_dims(np.expand_dims(np.expand_dims(mfcc_feat, axis=0), axis=0), axis=0)

    cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())
    print(cct.shape)

print("* done")

stream.stop_stream()
stream.close()
p.terminate()
