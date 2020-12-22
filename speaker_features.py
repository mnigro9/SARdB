#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 17:23:53 2020

@author: Admin

Making features for speaker counting based on literature approaches:
    
    ∆ Wei, Determining number of speakers.... spectrograms, 8kHz stream uses 
    STFT frame size 25ms with 50% overlpa. 49 x 94 spectrogram image size
    
    Stoter, CountNet.... absolute value of STFT with Hann windows, 25ms frame 
    size and 10ms hop size. 5s samples at 16kHz produced 500x201 TF representation.
    also applied log(1+STFT) and mel40. 
    
    Zhang, End to end overlapped speech detection .... 
    500ms windows of speech, input to NN is 8000 dimensional raw waveform
    
    Valentin, OVerlapped speech....flattened spectrogram,signal envelope, and hisotgram 
    of speech signal as input
"""

import librosa
import csv
import numpy as np
import glob
#%% Load filenames and save for future use the order they appear as txt files
train = sorted(glob.glob('SAR_v4/train*.wav'))
test = sorted(glob.glob('SAR_v4/test*.wav'))
valid = sorted(glob.glob('SAR_v4/valid*.wav'))
#%%
with open('training_dir_SAR_v4.txt','w') as f:
    f.write('\n'.join(train))

with open('testing_dir_SAR_v4.txt','w') as f:
    f.write('\n'.join(test))
    
with open('validation_dir_SAR_v4.txt','w') as f:
    f.write('\n'.join(valid))    
#%% Functions for feature computation    
def abstft(train,nf,hop):  #input list of filenames, fft size, hop length in samples
    '''
    produces the magnitude STFT of a signal and speaker count label
    '''
    dat = []
    label = np.zeros((len(train),1))

    for i,filename in enumerate(train):
        y,sr = librosa.load(filename,sr=16000)
        #num rows is 1+n_fft/2 and num cols is 
        m = np.abs(librosa.stft(y,n_fft=nf,hop_length=hop))   #TRY: 50ms window...800 samples
        dat.append(m)
        label[i] = int(filename[-7])
    return dat,label

# spectrogram = abs(STFT)^2
def spec(dat):  #input list of abs(STFT)
    ''' produces the spectrogam of a given |STFT|'''
    sp = []
    for d in dat:
        sp.append(d**2)
    return sp
#%%
#train_stft,train_label = abstft(train,800,400)  #50ms window, 50% overlap (800,400) yileds 401x401 matrix
train_stft25,train_label = abstft(train,400,200)  #25ms window, 50% overlap (400,200) yields 201x801 matrix

train_spec25 = spec(train_stft25)

np.savez('train_tf25',train_stft=train_stft25,train_spec=train_spec25,train_label=train_label)    

#%% Mel log spectrogram
'''
parameters used: sr=44100, window=2048, hop=511, mels=64, max len seconds=10, max_frames= math.ceil(max_len_seconds * sr / hop), fmin=0, fmax=22050

#### use a 50ms window (2019baseline), 11.58ms hop and mel=64 or 100ms (~93ms) window(2019 2nd place) with hop=16.55ms, mels=128)
'''
#%% to read .txt file with list of filenames for dataset splits. only need if loading from file
import numpy as np

train_dir = []
test_dir=[]  #load the file paths
with open('training_dir2.txt','r') as f:  
  for line in f:
    train_dir.append(line.rstrip('\n'))  #remove new line char
with open('testing_dir2.txt','r') as f:
  for line in f:
    test_dir.append(line.rstrip('\n'))

print('managed reading files \n')
#%%
''' function to get the mel scale spectrogram from a ist of audio files.
    inpit: file list, fft size, hop size, and number of mel bands
    output: melscale spectrogram, speaker count label, sound event count label 
'''    
def get_melspec(train,nfft,hop,nmel):
    
  f_min = 0
  f_max = 8000  #sr/2
  dat=[]
  label = np.zeros((len(train),1))
  labele = np.zeros((len(train),1))

  #p = 'drive/My Drive/Colab Notebooks/'
  for i,file in enumerate(train):
    y,sr = librosa.load(file,sr=16000)
    label[i] = int(file[-7])
    labele[i] = int(file[-5])
    spec = librosa.stft(y,n_fft=nfft,hop_length=hop,window='hamm',center=True,pad_mode='reflect')

    mel_spec = librosa.feature.melspectrogram(
            S=np.abs(spec),  # amplitude, for energy: spec**2 but don't forget to change amplitude_to_db.
            sr=sr,
            n_mels=nmel,
            fmin=f_min, fmax=f_max,
            htk=False, norm=None)

    mel_log_spec = librosa.amplitude_to_db(mel_spec) # 10*log10(S**2 / ref), ref=1 is default

    mel_log_spec = mel_log_spec.T
    dat.append(mel_log_spec)
  return dat,label,labele
'''
train_mel = get_melspec(train_dir,1024,256,64)  #50ms window (800 ~1024 samples), 10ms hop (160 ~ 256 samples), 64 mels
test_mel = get_melspec(test_dir,1024,256,64)  #50ms window (800 ~1024 samples), 10ms hop (160 ~ 256 samples), 64 mels
'''
train_mel,train_label,train_sfx = get_melspec(train,256,128,64)  #100ms window (1600 ~2048 samples), 16ms hop (256 samples), 128 mels
test_mel,test_label,test_sfx = get_melspec(test,256,128,64)  
valid_mel,valid_label,valid_sfx = get_melspec(valid,256,128,64)
print('got mel features \n')

'''np.savez('trainfull_mel64_10ms',train_mel=train_mel)
np.savez('testfull_mel64_10ms',test_mel=test_mel)'''
np.savez('SAR_v4_mel64_256fft_128hop',train_mel=train_mel,valid_mel=valid_mel,test_mel=test_mel,
         train_label = train_label, valid_label = valid_label, test_label=test_label,
         train_sfx = train_sfx, valid_sfx = valid_sfx, test_sfx = test_sfx)
print('saved')

#%%
########################
########################
########################3#######################################################
########################3#######################################################
########################3#######################################################
########################3#######################################################
########################3#######################################################
#%% Alice TFsqueezed and Emphasis transformation
import tfutil
import numpy as np
import librosa

train_dir = []
test_dir=[]  #load the file paths
with open('training_dir.txt','r') as f:  
  for line in f:
    train_dir.append(line.rstrip('\n'))  #remove new line char

db_freqSqueezedPSD=[]
# Processing signal
for addr in train_dir:
  audio, fs = librosa.load(addr,sr=16000) #sf.read(addr)
  _, _, psd, _ = tfutil.createPSD(audio,fs)
  truncNormPSD = tfutil.truncatePSD(psd)
  timeSqueezedPSD = tfutil.groupTimePSD(truncNormPSD, N=4)
  freqSqueezedPSD = tfutil.groupFreqPSD(timeSqueezedPSD, Nupper=5)
  freqSqueezedPSD[freqSqueezedPSD==0] = np.finfo(float).eps
  db_freqSqueezedPSD.append(10*np.log10(freqSqueezedPSD))

np.savez('dbSingleEmphasis_train',train = db_freqSqueezedPSD)  
#%%
import matplotlib.pyplot as plt
import librosa.display

plt.subplot(4,2,1)        
librosa.display.specshow(db_freqSqueezedPSD[0], y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram-1')
plt.subplot(4,2,2)        
librosa.display.specshow(db_freqSqueezedPSD[1], y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram-1')
plt.subplot(4,2,3)
librosa.display.specshow(db_freqSqueezedPSD[11], y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram-2')
plt.subplot(4,2,4)
librosa.display.specshow(db_freqSqueezedPSD[13], y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram-2')
plt.subplot(4,2,5)
librosa.display.specshow(db_freqSqueezedPSD[3], y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram-3')
plt.subplot(4,2,6)
librosa.display.specshow(db_freqSqueezedPSD[2], y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram-3')
plt.subplot(4,2,7)
librosa.display.specshow(db_freqSqueezedPSD[14], y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram-4')
plt.subplot(4,2,8)
librosa.display.specshow(db_freqSqueezedPSD[15], y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram-4')

plt.show()
#%% rank estimation. use each row of TF to make covariance matrix for sample
C=[]
eigens=[]
avg_E=[]
for i,file in enumerate(db_freqSqueezedPSD):
    A = np.cov(db_freqSqueezedPSD[i].T)
# eigenvalues
    E,_ = np.linalg.eig(A)
    eigens.append(E)
    avg_E.append(np.mean(E))
#% rank sourc count
    C.append(np.linalg.matrix_rank(E,0.005))  #0.15
