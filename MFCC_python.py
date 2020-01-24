import time
from Options_all import BaseOptions
from util import util
from util.visualizer import Visualizer
from torch.utils.data import DataLoader
import os
import ntpath
import cv2
import torch
import util.util as util
import numpy as np
import pyaudio
import wave
from python_speech_features import mfcc
import copy
import runmfcc
import Test_Gen_Models.Test_Audio_Model as Gen_Model
from Dataloader.Test_load_audio import Test_VideoFolder

RATE = 16000
t = time.time()
wf = wave.open('trump.wav', 'rb')
wav = wf.readframes(RATE*500)

print(wav)
wav = np.fromstring(wav, 'Int16')
mfcc_feat = mfcc(wav, RATE, winlen=0.025, winstep=0.01, numcep=13,
                          nfilt=13, nfft=512, lowfreq=300, highfreq=3700, preemph=0.97, ceplifter=22,
                          appendEnergy=True)
mfcc_feat = np.transpose(mfcc_feat)
index = 20*int(mfcc_feat.shape[1]/20)
mfcc_feat = mfcc_feat[1:,:index]
print(mfcc_feat.shape)
mfcc_feat = np.array(np.split(mfcc_feat, 20,axis=1))
mfcc_feat = np.transpose(mfcc_feat)
print(time.time()-t)
print(mfcc_feat.shape)