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


mat = runmfcc.initialize()
opt = BaseOptions().parse()



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
        noi = 1000.0 * np.random.normal(0, 0.1, CHUNK)
        noi = noi.astype(np.int16)
        sig = copy.deepcopy(wav[:CHUNK])+noi
        wav = np.concatenate((wav[CHUNK:],wav[:CHUNK]))
    out_data = copy.deepcopy(sig)


    sig = np.concatenate((streamedAudio, sig))

    if sig.shape[0] > int(RATE*0.2*10):
        streamedAudio = sig[int(-1.0*RATE*0.2*10):]
    else:
        streamedAudio = sig
    print(len(streamedAudio))
    mfcc_matlab = mat.runmfcc(sig.tolist())
    mfcc_matlab = np.array(mfcc_matlab)
    mfcc_matlab = np.transpose(mfcc_matlab)[-20:,1:]

    '''mfcc_feat = mfcc(sig, RATE, winlen=0.025, winstep=0.01, numcep=13,
                          nfilt=13, nfft=2048, lowfreq=300, highfreq=3700, preemph=0.97, ceplifter=22,
                          appendEnergy=True)
    mfcc_feat = np.transpose(mfcc_feat)'''
    mfcc_feat = mfcc_matlab
    audio_set = np.expand_dims(np.expand_dims(np.expand_dims(mfcc_feat, axis=0), axis=0), axis=0)
    mfcc_data = np.array(audio_set)
    return (out_data, pyaudio.paContinue)



mode = 1 # 1=ours, 2=benchmark, 3=combined
CHUNK = 10000#4410
WIDTH = 2
CHANNELS = 1
RATE = 44100
t = time.time()
mfcc_data = []
p = pyaudio.PyAudio()
n = 1



if mode == 3:
    wf = wave.open('trump.wav', 'rb')
    wav = wf.readframes(RATE*500)
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
test_nums = [7]
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
            #print(data['B_audio'])
            model.set_test_input(data)
            model.test()
            visuals = model.get_current_visuals()
            im_rgb = cv2.cvtColor(visuals['fake_audio_B_0'], cv2.COLOR_RGB2BGR)
            cv2.imshow('TEST', im_rgb)
            cv2.waitKey(1)
        end = time.time()
        print('GENERATE',end - start,'sec')




