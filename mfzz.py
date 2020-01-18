import numpy as np


def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n

def mfcc(speech, fs, Tw, Ts, alpha, window, R, M, N, L ):


    if (np.max(np.abs(speech)) <= 1):
        speech = speech*pow(2,15)

    Nw = round(Tw*fs/1000)
    Ns = round(Ts*fs)

    nfft = 2^nextpow2(Nw)
    K = nfft/2+1

    print(K)


x = np.array([1,2,3,4,5,6])
y = mfcc(x,16000,25,10,0.97,1,1,1,1,1)
