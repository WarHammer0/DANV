function [CC] = runmfcc( speech )
    
    %speech = cell2mat(speech);
    fs = 44100;
    Tw = 25;
    Ts = 10;
    alpha = 0.97;
    R = [300 3700];
    M = 13;
    C = 13;
    L = 22;
    N = C ;
    hamming = @(N)(0.54-0.46*cos(2*pi*[0:N-1].'/(N-1)));
    [ CC, FBE, frames ] = mfcc( speech, fs, Tw, Ts, alpha, hamming, R, M, N, L );
end