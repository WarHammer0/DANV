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

opt = BaseOptions().parse()
import Test_Gen_Models.Test_Audio_Model as Gen_Model
from Dataloader.Test_load_audio import Test_VideoFolder


# global variable
streamedAudio = np.zeros(8820,dtype=np.int16)
audio_set = np.array([0.0])


def callback(in_data, frame_count, time_info, status):
    global streamedAudio
    global mfcc_data
    global wav
    global RATE
    global mode

    if mode != 3:
        sig = np.fromstring(in_data, 'Int16')
    else:
        sig = copy.deepcopy(wav[:CHUNK])+np.fromstring(in_data, 'Int16')
        wav = np.concatenate((wav[CHUNK:],wav[:CHUNK]))
    sig = np.concatenate((streamedAudio, sig))

    if sig.shape[0] > int(RATE*0.2):
        streamedAudio = sig[int(-1.0*RATE*0.2):]
    else:
        streamedAudio = sig

    mfcc_feat = zip(*mfcc(sig[:int(RATE*0.2)], RATE, winlen=0.025, winstep=0.0095, numcep=12,
                          nfilt=13, nfft=2048, lowfreq=300, highfreq=3700, preemph=0.97, ceplifter=22,
                          appendEnergy=True))
    mfcc_feat = np.stack([np.array(i) for i in mfcc_feat])
    audio_set = np.expand_dims(np.expand_dims(np.expand_dims(mfcc_feat, axis=0), axis=0), axis=0)
    mfcc_data = np.array(audio_set)
    return (in_data, pyaudio.paContinue)



mode = 3 # 1=ours, 2=benchmark, 3=combined
CHUNK = 1600#4410
WIDTH = 2
CHANNELS = 1
RATE = 16000#44100
t = time.time()
mfcc_data = []
p = pyaudio.PyAudio()
n = 1



if mode == 3:
    wf = wave.open('0572_0019_0003.wav', 'rb')
    wav = wf.readframes(16000*4)
    wav = wav[1::2]
    wav = np.fromstring(wav, 'Int16')

stream = p.open(format=pyaudio.paInt16,  # p.get_format_from_width(WIDTH),
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK,
                stream_callback=callback)


opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.sequence_length = 1
test_nums = [3]
model = Gen_Model.GenModel(opt)

A_path = os.path.join(opt.test_A_path, 'test_sample' + str(test_nums[0]) + '.jpg')
test_folder = Test_VideoFolder(root='./0572_0019_0003', A_path=A_path, config=opt)
test_dataloader = DataLoader(test_folder, batch_size=1)

model, _, start_epoch = util.load_test_checkpoint('./checkpoints/101_DAVS_checkpoint.pth.tar', model)

# inference during test
for i2, data in enumerate(test_dataloader):
    if i2 < 5:
        # data['A'] = dic['A']
        model.set_test_input(data)
        model.test_train()
    else:
        break

enum = list(enumerate(test_dataloader))
print("* recording")
while(1):
    if mode == 2:
        k = 99
    else:
        k = 1
    for i in range(0,k):
        start = time.time()
        mfcc_data_torch = torch.autograd.Variable(torch.from_numpy(np.array(mfcc_data)).float())

        data = {}
        dic = enum[i][1]
        data['A'] = dic['A']
        data['A_path'] = dic['A_path']

        if 1:
            if mode == 2:
                data['B_audio'] = dic['B_audio']
            else:
                data['B_audio'] = mfcc_data_torch[0]
            model.set_test_input(data)
            model.test()
            visuals = model.get_current_visuals()
            im_rgb = cv2.cvtColor(visuals['fake_audio_B_0'], cv2.COLOR_RGB2BGR)
            cv2.imshow('TEST', im_rgb)
            cv2.waitKey(1)
        end = time.time()
        print('GENERATE',end - start,'sec')




