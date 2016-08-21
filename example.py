#!/usr/bin/env python

import sys
sys.path.append('python_speech_features')
from base import mfcc, logfbank
#from base import logfbank
import scipy.io.wavfile as wav

(rate,sig) = wav.read("english.wav")
mfcc_feat = mfcc(sig,rate)
fbank_feat = logfbank(sig,rate)

print(fbank_feat[1:3,:])
