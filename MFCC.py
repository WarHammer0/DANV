from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy
import os
import torch

# directory where we your .wav files are
from scipy.io import wavfile

directoryName = '''C:/Users/Ryuusei/PycharmProjects/untitled1/DAVS/sound'''  # put your own directory here
# directory to put our results in, you can change the name if you like
resultsDirectory = directoryName + "/MFCCresults"

# make a new folder in this directory to save our results in
if not os.path.exists(resultsDirectory):
    os.makedirs(resultsDirectory)

# get MFCCs for every .wav file in our specified directory
for filename in os.listdir(directoryName):
    if filename.endswith('put.wav'):  # only get MFCCs from .wav s
        # read in our file
        (rate, sig) = wav.read(directoryName + "/" + filename)
        print(sig[5000:5000+int(44100*0.2)])
        print(sig.shape,rate)
        # get mfcc

        mfcc_feat = zip(*mfcc(sig[1000:1000+int(44100*0.2)],
                              rate,
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
        mfcc_feat = numpy.stack([numpy.array(i) for i in mfcc_feat])
        print(mfcc_feat.shape)

        cc = numpy.expand_dims(numpy.expand_dims(mfcc_feat, axis=0), axis=0)
        # print([cc])
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

