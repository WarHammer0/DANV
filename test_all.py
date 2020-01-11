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
from python_speech_features import mfcc
import copy

opt = BaseOptions().parse()
import Test_Gen_Models.Test_Audio_Model as Gen_Model
from Dataloader.Test_load_audio import Test_VideoFolder

def callback(in_data, frame_count, time_info, status):
    print(in_data)
    return (data, pyaudio.paContinue)

def MFCC_realtime(mfcc_num,streamedAudio):

    CHUNK = 4410
    WIDTH = 2
    CHANNELS = 1
    RATE = 44100
    t = time.time()
    audio_set = []
    p = pyaudio.PyAudio()
    n = mfcc_num

    stream = p.open(format=pyaudio.paInt16,#p.get_format_from_width(WIDTH),
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=callback)

    data = stream.read(CHUNK)
    sig = np.fromstring(data, 'Int16')
    sig = np.concatenate((streamedAudio,sig))
    stream.write(sig, CHUNK)

    if sig.shape[0] > 8820:
        streamedAudio = sig[-8820:]
    else:
        streamedAudio = sig



    for i in range(0, mfcc_num,1 ):
        step = (len(sig)-8820)/mfcc_num
        mfcc_feat = zip(*mfcc(sig[int(i*step):int(i*step+8820)],
                              RATE,
                              winlen=0.025,
                              winstep=0.0095,
                              numcep=12,
                              nfilt=13,
                              nfft=2048,
                              lowfreq=300,
                              highfreq=3700,
                              preemph=0.97,
                              ceplifter=22,
                              appendEnergy=True))
        mfcc_feat = np.stack([np.array(i) for i in mfcc_feat])

        cc = np.expand_dims(np.expand_dims(np.expand_dims(mfcc_feat, axis=0), axis=0), axis=0)

        '''cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())
        shape = cct.size()
        if shape != torch.Size([1, 1, 1, 12, 20]):
            print(shape)

        elif shape == torch.Size([1, 1, 1, 12, 20]):'''
        audio_set.append(cc)
    audio_set = np.reshape(audio_set,[-1,1,1,12,20])
    print('STREAM',time.time()-t, 'sec')
    return np.array(audio_set),streamedAudio

opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.sequence_length = 1
mfcc_num = 1
test_nums = [2]
model = Gen_Model.GenModel(opt)
streamedAudio = np.zeros(8820,dtype=np.int16)


#_, _, start_epoch = util.load_test_checkpoint(opt.test_resume_path, model)
start_epoch = opt.start_epoch
visualizer = Visualizer(opt)
# find the checkpoint's path name without the 'checkpoint.pth.tar'
path_name = ntpath.basename(opt.test_resume_path)[:-19]
if True:
    for i in test_nums:
        A_path = os.path.join(opt.test_A_path, 'test_sample' + str(i) + '.jpg')

        test_folder = Test_VideoFolder(root='./0572_0019_0003', A_path=A_path, config=opt)
        test_dataloader = DataLoader(test_folder, batch_size=1)

        model, _, start_epoch = util.load_test_checkpoint('./checkpoints/101_DAVS_checkpoint.pth.tar', model)

        # inference during test
        for i2, data in enumerate(test_dataloader):
            if i2 < 5:
                # data['A'] = dic['A']
                model.set_test_input(data)
                model.test_train()

        k = 0
        enum = enumerate(test_dataloader)

        while(True):

            print("* recording")

            dict = list(enum)
            dic = dict[0][1]

            k+=1
            if k >= len(list(enum)):
                k = 0
            mfcc_data,streamedAudio = MFCC_realtime(mfcc_num,streamedAudio)
            start = time.time()



            data = {}
            data['A'] = dic['A']
            data['A_path'] = dic['A_path']
            mfcc_data_torch = torch.autograd.Variable(torch.from_numpy(mfcc_data).float())


            for j in range(0,int(mfcc_num)):
                data['B_audio'] = dic['B_audio']#mfcc_data_torch[j]
                print(data['B_audio'])
                model.set_test_input(data)
                model.test()
                visuals = model.get_current_visuals()
                #img_path = model.get_image_paths()
                im_rgb = cv2.cvtColor(visuals['fake_audio_B_0'], cv2.COLOR_RGB2BGR)
                cv2.imshow('TEST', im_rgb)
                cv2.waitKey(1)
            end = time.time()
            print('GENERATE',end - start,'sec')







