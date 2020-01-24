from HTKFeat import MFCC_HTK
from HTK import HCopy, HTKFile
import numpy as np


class mfcc_htk():

    def __init__(self,fs):

        self.win_shift = int(0.01*fs)
        self.win_len = int(0.025*fs)
        self.k = 0.97
        self.filter_num = 13
        self.mfcc_num = 13
        self.lifter_num = 22

        self.mfcc = MFCC_HTK(samp_freq=fs,win_len=self.win_len,win_shift=self.win_shift,preemph=self.k,
                             filter_num=self.filter_num,mfcc_num=self.mfcc_num,lifter_num=self.lifter_num)


        self.h = [1, -self.k]
        self.rect = np.ones(self.win_len)
        self.hamm = np.hamming(self.win_len)
        self.f = np.linspace(0, 8000, 1000)
        freq2mel = lambda freq: 1127 * (np.log(1 + ((freq) / 700.0)))
        self.m = freq2mel(self.f)
        self.mfcc.create_filter(self.filter_num)
        #self.mfcc.load_filter('filter.csv')
        self.mfnorm = np.sqrt(2.0 / self.filter_num)

    def run(self,signal):
        signal = signal - np.mean(signal)
        sig_len = len(signal)

        # this is how we compute the window number while discarding the ones that don't fit the signal completely
        win_num = np.floor((sig_len - self.win_len) / self.win_shift).astype('int') + 1

        wins = []
        for w in range(win_num):
            # these are the start and end of each window
            s = w * self.win_shift
            e = s + self.win_len

            # we need the copy of the data, because the numpy array slicing gives us a view of the data
            # and we don't won't to mess up the original signal when we start modifying the windows
            win = signal[s:e].copy()

            wins.append(win)

        wins = np.asarray(wins)
        for win in wins:
            win-=np.hstack((win[0],win[:-1]))*self.k


        for win in wins:
            win*=self.hamm

        fft_len=np.asscalar(2**(np.floor(np.log2(self.win_len))+1).astype('int'))

        ffts=[]
        for win in wins:
            win=np.abs(np.fft.rfft(win,n=fft_len)[:-1])
            ffts.append(win)

        ffts=np.asarray(ffts)


        melspec=[]
        for f in ffts:
            m = np.dot(f,self.mfcc.filter_mat)
            melspec.append(m)
        melspec=np.asarray(melspec)

        melspec = np.log(melspec)

        dct_base = np.zeros((self.filter_num, self.mfcc_num));
        for m in range(self.mfcc_num):
            dct_base[:, m] = np.cos((m + 1) * np.pi / self.filter_num * (np.arange(self.filter_num) + 0.5))

        mfccs=[]
        for m in melspec:
            c=np.dot(m,dct_base)
            mfccs.append(c)
        mfccs=np.asarray(mfccs)
        mfnorm = np.sqrt(2.0 / self.filter_num)
        mfccs*=mfnorm
        lifter=1+(self.lifter_num/2)*np.sin(np.pi*(1+np.arange(self.mfcc_num))/self.lifter_num)
        mfccs*=lifter
        return mfccs

'''htk = mfcc_htk()
wf = wave.open('00001.wav', 'rb')
wav = wf.readframes(wf.getframerate() * wf.getnframes())
wav = np.fromstring(wav, 'Int16')
mfcc = htk.run(wav)
P.figure(figsize=(15,5))
P.pcolormesh(mfcc.T,cmap='gray')
P.show()'''



