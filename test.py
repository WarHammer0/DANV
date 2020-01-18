
from python_speech_features import mfcc
import numpy as np
import wave
import matplotlib.pyplot as plt
from matplotlib import cm
from torch.utils.data import DataLoader
from Dataloader.Test_load_audio import Test_VideoFolder
import os
from Options_all import BaseOptions
import matlab.engine
matlab = matlab.engine.start_matlab()


wf = wave.open('0572_0019_0003.wav', 'rb')
wav = wf.readframes(16000*10)
wav = wav[1::2]
wav = np.fromstring(wav, 'Int16')
wav = wav




opt = BaseOptions().parse()
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.sequence_length = 1
A_path = os.path.join(opt.test_A_path, 'test_sample' + str(3) + '.jpg')
test_folder = Test_VideoFolder(root='./0572_0019_0003', A_path=A_path, config=opt)
test_dataloader = DataLoader(test_folder, batch_size=1)

enum = list(enumerate(test_dataloader))

mfcc_bin = enum[0][1]['B_audio'].numpy()[0][0][0]

mfcc_feat = mfcc(wav[0:10000], 16000, winlen=0.025, winstep=0.01, numcep=13,
                      nfilt=13, nfft=16*25, lowfreq=300, highfreq=3700, preemph=0.97, ceplifter=22,
                      appendEnergy=True)

mfcc_matlab = matlab.runmfcc(wav.tolist())
mfcc_matlab = np.array(mfcc_matlab)
mfcc_matlab = np.transpose(mfcc_matlab)+1.2
print(mfcc_matlab.shape)
mfcc_feat = np.transpose(mfcc_feat)[1:,:20]
audio_set = np.expand_dims(np.expand_dims(np.expand_dims(mfcc_feat, axis=0), axis=0), axis=0)

ig, ax = plt.subplots()
cax = ax.imshow(mfcc_bin, interpolation='nearest', cmap=cm.coolwarm, origin='lower', aspect='auto')
print(np.mean(mfcc_bin))
ax.set_title('MFCC')
#Showing mfcc_data

ig2, ax2 = plt.subplots()
mfcc_data= np.swapaxes(mfcc_feat, 0 ,1)
cax2 = ax2.imshow(mfcc_matlab[0:20,1:], interpolation='nearest', cmap=cm.coolwarm, origin='lower', aspect='auto')
print(np.mean(mfcc_matlab[0:20,1:]))
ax2.set_title('MFCC')
#Showing mfcc_data
plt.show()

#audio_set = np.expand_dims(np.expand_dims(np.expand_dims(mfcc_feat, axis=0), axis=0), axis=0)
#mfcc_data = np.array(audio_set)