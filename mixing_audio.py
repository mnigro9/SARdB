#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 22:22:35 2020

@author: Admin

Mixing audio files of speech with non-speech sources.

1. Make the signals the same length, same number of channels. 
1a. The non-speech signal should be padded with zeros randomly at beginning and end.
1b. Non-speech should be low dB so speech is heard clearly, but sound is distinct.
2. Add and divide by 2
"""
#%% sound effect sorting. don't need anymore.
import csv
import numpy as np
vehicle=[]
ambulances=[]
alarms=[]
horns=[]
with open("/Users/Admin/Downloads/annotations-dev.csv") as f:
    ann = csv.reader(f)
    next(ann) #skip header row
    for row in ann:
        sirens = np.array(list(map(float,row[18:22])))  #vehicle sound tags
        ambulance = float(row[20]) 
        horn = float(row[18]) 
        alarm = float(row[19]) 
        cats = np.array(list(map(float,row[4:32]))) #all categories 18-22 vehicle sound tags
        if sum(cats)==1 and sum(sirens)==1: # and row[0]=='train':
            vehicle.append(row[2]) #filename
        if sum(cats)==1 and ambulance==1:
            ambulances.append(row[2])
        if sum(cats)==1 and horn==1:
            horns.append(row[2])
        if sum(cats)==1 and alarm==1:
            alarms.append(row[2])
            
uni_files = list(set(vehicle))        
u_ambulances = list(set(ambulances))  #get unique values in list
u_alarms = list(set(alarms))
u_horns = list(set(horns))
#%% sound effects. get training and testing split

# Parse through sound effects files
import glob
import random
import numpy as np
import soundfile as sf
import librosa

doors = sorted(glob.glob('/Users/Admin/Downloads/dcase2016_task2_train_dev/dcase2016_task2_train/doorslam*.wav'))
doors_tr = doors[0:10]
doors_v = doors[10:15]
doors_te = doors[-5:]

phone = sorted(glob.glob('/Users/Admin/Downloads/dcase2016_task2_train_dev/dcase2016_task2_train/phone*.wav'))
phone_tr = phone[0:10]
phone_v = phone[10:15]
phone_te = phone[-5:]

keyboard = sorted(glob.glob('/Users/Admin/Downloads/dcase2016_task2_train_dev/dcase2016_task2_train/keyboard*.wav'))
keyboard_tr = keyboard[0:10]
keyboard_v = keyboard[10:15]
keyboard_te = keyboard[-5:]

knocking = sorted(glob.glob('/Users/Admin/Downloads/dcase2016_task2_train_dev/dcase2016_task2_train/knock*.wav'))
knocking_tr = knocking[0:10]
knocking_v = knocking[10:15]
knocking_te = knocking[-5:]

sirens_tr = ['/Users/Admin/Downloads/audio-dev/train/27_002114.wav','/Users/Admin/Downloads/audio-dev/train/25_001078.wav','/Users/Admin/Downloads/audio-dev/train/34_002331.wav','/Users/Admin/Downloads/audio-dev/train/34_000993.wav',
          '/Users/Admin/Downloads/audio-dev/validate/00_000277.wav']
sirens_v = ['/Users/Admin/Downloads/audio-dev/validate/03_000076.wav',
          '/Users/Admin/Downloads/audio-dev/train/36_000809.wav']
sirens_te = ['/Users/Admin/Downloads/audio-dev/validate/03_002315.wav',
          '/Users/Admin/Downloads/audio-dev/train/38_000322.wav',
          '/Users/Admin/Downloads/audio-dev/validate/03_001150.wav']

honks_tr = ['/Users/Admin/Downloads/audio-dev/train/11_001288.wav','/Users/Admin/Downloads/audio-dev/validate/03_001109.wav',
         '/Users/Admin/Downloads/audio-dev/train/32_001593.wav','/Users/Admin/Downloads/audio-dev/train/40_000540.wav',
         '/Users/Admin/Downloads/audio-dev/train/02_002533.wav']
honks_v = ['/Users/Admin/Downloads/audio-dev/train/40_000049.wav',
         '/Users/Admin/Downloads/audio-dev/train/37_002073.wav','/Users/Admin/Downloads/audio-dev/train/18_000642.wav']
honks_te = ['/Users/Admin/Downloads/audio-dev/train/40_002369.wav','/Users/Admin/Downloads/audio-dev/train/25_001249.wav']

outdoors_tr = sirens_tr+honks_tr  #merge the 2 lists. keep honks and sirens as full 10s 
outdoors_te = sirens_te+honks_te
outdoors_v = sirens_v+honks_v
# amplitude values for corresponding sound effects
d = [0.2,0.3,0.4,0.5] #doors
p = [0.05,0.1,0.15,0.2] #phone
k = [0.5,0.65,0.8,0.95]  #keyboard and knocking
n = [0.5,0.65,0.8,0.95]  #knocking
o = [0.01,0.04,0.06,0.08] #outdoors

'''

#############################################

NAMING CONVENTION FOR SFX: dxxxx means 1 sfx - doorslam
where d(doorslam), p(phone), k(keyboard), n(knocking), o(outdoor)

xxxxx means no sfx included
'''
#%%
from pydub import AudioSegment
from pydub.playback import play
myaudio = AudioSegment.from_file(phone_te[0],"wav",frame_rate=16000)  #load one file
#play(myaudio)
#%%
newaudio = myaudio - 20  #db dec
newaudio.export(out_f=
"louder_wav_file20db.wav"
, format=
"wav")
#%%
y,sr = librosa.load(phone_te[0],sr=16000)
ynew = y*0.8
ydb = y*0.08
#sf.write('08a.wav',ynew,sr,'PCM_16') 
y2,sr = librosa.load('louder_wav_file20db.wav',sr=16000)
#%% functions
def single_sound(signal,A,sig_length):  # create a 10s segment of a single office sound  
    #import matplotlib.pyplot as plt
    
    temp = np.zeros(sig_length,dtype=np.float16)  #16khz x 10s = 160,000 samples
    startpoint = np.random.randint(sig_length-len(signal))
    endpoint = startpoint+len(signal)
    temp[startpoint:endpoint] = A*signal  #A: amplitude adjustment value. Possibly break this into 2 steps, adding to temp than amplitude multiply
    '''    plt.subplot(2,1,1)
    plt.plot(signal)
    plt.title('orig')plt.subplot(2,1,2)
    plt.plot(temp)
    plt.show()'''
    
    return temp,startpoint,endpoint


def amplitude_set(sfx):  #determine the appropriate amplitude value
    ### *0.1 equivalent to about 20dB decrease. 0.3 equivalent ~10dB
    d = [0.2,0.3,0.4,0.5]#[1,1,1,1,1]#[0.2,0.3,0.4,0.5] #doors
    p = [0.05,0.1,0.15,0.2]#[0.1,0.12,0.15,0.17,0.2] #[0.05,0.1,0.15,0.2] #phone
    k = [0.5,0.65,0.8,0.95]#[1,1,1,1,1]#[0.5,0.65,0.8,0.95]  #keyboard and knocking
    n = [0.5,0.65,0.8,0.95]#[1,1,1,1,1]#[0.5,0.65,0.8,0.95]  #keyboard and knocking
    o = [0.01,0.04,0.06,0.08]#[0.1,0.12,0.15,0.17,0.2]#[0.01,0.04,0.06,0.08] #outdoors
    if 'door' in sfx:
        amp = random.choice(d)  #amplitudes for doorslam
        l = 'd'
    elif 'phone' in sfx:
        amp = random.choice(p)  #amplitude for phone ring
        l = 'p'
    elif 'audio' in sfx:
        amp = random.choice(o)  #amplitude for outdoors
        l = 'o'
    elif 'knock' in sfx:
        amp = random.choice(n)  #amplitude for knock or keyboard
        l = 'n'
    else:
        amp = random.choice(k) #keyboad
        l = 'k'
    return amp,l
    
def mix_speech_train(speechfiles,sfxtrain,sp_num,total,folder):  
    #1. pass glob for 1 spk, then 2 spk, 3, 4 training file. Use 2250 files in training per speaker class
    #2 : the master sound effect training file list of lists
    #3: number of speakers 1,2,3 or 4
    #4: number of samples 
    #5: folder to save data (string) 'SARfull/'
    import math
    from itertools import combinations
    #total = 3012 #len(speechfiles)  
    splitter = math.floor(total/6)  #making the number of sound effects equally distributed in the speaker class.
    
    flat_sfxtrain = [item for sublist in sfxtrain for item in sublist]  #flatten. 75 dimensional
    length_sfx = len(sfxtrain[0])  #number of files per sfx type
    tok=0
    usedfiles=[]
    #sig_length = 160000
    for i, speech in enumerate(speechfiles):
        
        if i>total-1: #2249:  #for uniform distribution of samples per class
            break
        spk,fs = librosa.load(speech,sr=16000)
        sig_length = len(spk)  
        title=speech[14:19]          
        #name = speech[19:-6]
        #name = speech[-23:-6]
        name = speech[20:-6]
        if i<splitter:# i<375:  #1 sound effect
            so_num = 1
            seffect = flat_sfxtrain[tok]  #sound effect file name
            #spk,fs = librosa.load(speech,sr=16000)
            sfx,fs1=librosa.load(seffect,sr=16000)
            amp,l = amplitude_set(seffect)   
            if len(sfx)<sig_length:
                sound,sp,ep = single_sound(sfx,amp,sig_length)
            else:
                sound = sfx*amp  #outdoor sound effect
                sp=0
                ep=sig_length
            mix = spk+sound
            savename = f'{title}_{name}_{l}xxxx_{sp_num}_{so_num}'
            print(savename)
            detail = f'{sp} {ep} {amp}'
            usedfiles.append([speech,seffect,str(sp_num), str(so_num),detail])#'1','1'])  #speaker file, sound effect file, num of speakers
            sf.write(folder+savename+'.wav',mix,fs,'PCM_16') 
            #sf.write(f'sound{name}.wav',sound,fs,'PCM_16')
            tok=tok+1
            if tok==len(flat_sfxtrain):#75:  #75 sound effect files in training split. change to 25 for test set
                tok=0  #start list from beginning. should result in equal representation of sound effect types 
        j=0
        if splitter-1<i<splitter*2:#374<i<750:  #2 random sounds. repeat process for 1 sound effect
            #spk,fs = librosa.load(speech,sr=16000)  #load speech file
            so_num = 2
            s2 = list(combinations([0,1,2,3,4],so_num))
            random.shuffle(s2)
            #ind_type = random.sample([0,1,2,3,4],so_num) #sfx type selection
            ind_file = random.sample(range(length_sfx),so_num) #sfx file selections
            
            #seffect1 = sfxtrain[ind_type[0]][ind_file[0]]  #get filenames
            #seffect2 = sfxtrain[ind_type[1]][ind_file[1]]
            seffect1 = sfxtrain[s2[j][0]][ind_file[0]]  #get filenames
            seffect2 = sfxtrain[s2[j][1]][ind_file[1]]

            amp1,l1 = amplitude_set(seffect1)
            amp2,l2 = amplitude_set(seffect2)
            
            sfx1,sam = librosa.load(seffect1,sr=16000)
            sfx2,sam = librosa.load(seffect2,sr=16000)
            
            if len(sfx1)<sig_length:
                sound1,sp1,ep1 = single_sound(sfx1,amp1,sig_length)
            else:
                sound1 = sfx1*amp1  #outdoor sound effect
                sp1=0
                ep1=sig_length
            
            if len(sfx2)<sig_length:
                sound2,sp2,ep2 = single_sound(sfx2,amp2,sig_length)
            else:
                sound2 = sfx2*amp2
                sp2=0
                ep2=sig_length
                
            mix = spk+sound1+sound2
            
            #savename = f'spk1_sfx2_{i}_12'  #
            savename = f'{title}_{name}_{l1}{l2}xxx_{sp_num}_{so_num}'  #Apr 28. naming convention changed to include sfx types used dkxxx
            print(savename)
            seffect = f'{seffect1} // {seffect2}'
            detail = f'{sp1} {ep1} {amp1} // {sp2} {ep2} {amp2}'
            usedfiles.append([speech,seffect,str(sp_num), str(so_num),detail])#'1','2'])  #speaker file, sound effect file, num of speakers, num of sounds
            sf.write(folder+savename+'.wav',mix,fs,'PCM_16') 
            if j==len(s2)-1:  #added this for permutations of sfx types. Apr 28
                j=0
            else:
                j=j+1
        j=0
        if splitter*2-1<i<splitter*3: #749<i<1125:  #sfx 3
            #spk,fs = librosa.load(speech,sr=16000)  #load speech file
            so_num = 3
            s3 = list(combinations([0,1,2,3,4],so_num))
            random.shuffle(s3)
            #ind_type = random.sample([0,1,2,3,4],so_num) #sfx type selection
            ind_file = random.sample(range(length_sfx),so_num) #sfx file selections
            
            seffect1 = sfxtrain[s3[j][0]][ind_file[0]]  #get filenames. Apr28 changed to reflect permutations of sfx combos
            seffect2 = sfxtrain[s3[j][1]][ind_file[1]]
            seffect3 = sfxtrain[s3[j][2]][ind_file[2]]
            
            amp1,l1 = amplitude_set(seffect1)
            amp2,l2 = amplitude_set(seffect2)
            amp3,l3 = amplitude_set(seffect3)
            
            sfx1,sam = librosa.load(seffect1,sr=16000)
            sfx2,sam = librosa.load(seffect2,sr=16000)
            sfx3,sam = librosa.load(seffect3,sr=16000)
            
            if len(sfx1)<sig_length:
                sound1,sp1,ep1 = single_sound(sfx1,amp1,sig_length)
            else:
                sound1 = sfx1*amp1  #outdoor sound effect
                sp1=0
                ep1=sig_length
            
            if len(sfx2)<sig_length:
                sound2,sp2,ep2 = single_sound(sfx2,amp2,sig_length)
            else:
                sound2 = sfx2*amp2
                sp2=0
                ep2=sig_length
            if len(sfx3)<sig_length:
                sound3,sp3,ep3 = single_sound(sfx3,amp3,sig_length)
            else:
                sound3 = sfx3*amp3
                sp3=0
                ep3=sig_length
                
            mix = spk+sound1+sound2+sound3
            
            #savename = f'spk1_sfx3_{i}_13'  #
            savename = f'{title}_{name}_{l1}{l2}{l3}xx_{sp_num}_{so_num}'
            print(savename)
            seffect = f'{seffect1} // {seffect2} // {seffect3}'
            detail = f'{sp1} {ep1} {amp1} // {sp2} {ep2} {amp2} // {sp3} {ep3} {amp3}'
            usedfiles.append([speech,seffect,str(sp_num), str(so_num),detail]) #'1','3'])  #speaker file, sound effect file, num of speakers, num of sounds
            sf.write(folder+savename+'.wav',mix,fs,'PCM_16') 
            if j==len(s3)-1:  #added this for permutations of sfx types. Apr 28
                j=0
            else:
                j=j+1        
        j=0
        if splitter*3-1<i<splitter*4: #1124<i<1500:  #sfx 4
            #spk,fs = librosa.load(speech,sr=16000)  #load speech file
            so_num = 4
            s4 = list(combinations([0,1,2,3,4],so_num))
            random.shuffle(s4)
            #ind_type = random.sample([0,1,2,3,4],so_num) #sfx type selection
            ind_file = random.sample(range(length_sfx),so_num) #sfx file selections
            
            seffect1 = sfxtrain[s4[j][0]][ind_file[0]]  #get filenames
            seffect2 = sfxtrain[s4[j][1]][ind_file[1]]
            seffect3 = sfxtrain[s4[j][2]][ind_file[2]]
            seffect4 = sfxtrain[s4[j][3]][ind_file[3]]
            
            amp1,l1 = amplitude_set(seffect1)
            amp2,l2 = amplitude_set(seffect2)
            amp3,l3 = amplitude_set(seffect3)
            amp4,l4 = amplitude_set(seffect4)
            
            sfx1,sam = librosa.load(seffect1,sr=16000)
            sfx2,sam = librosa.load(seffect2,sr=16000)
            sfx3,sam = librosa.load(seffect3,sr=16000)
            sfx4,sam = librosa.load(seffect4,sr=16000)
            
            if len(sfx1)<sig_length:
                sound1,sp1,ep1 = single_sound(sfx1,amp1,sig_length)
            else:
                sound1 = sfx1*amp1 #outdoor sound effect
                sp1=0
                ep1=sig_length
            
            if len(sfx2)<sig_length:
                sound2,sp2,ep2 = single_sound(sfx2,amp2,sig_length)
            else:
                sound2 = sfx2*amp2
                sp2=0
                ep2=sig_length
            if len(sfx3)<sig_length:
                sound3,sp3,ep3 = single_sound(sfx3,amp3,sig_length)
            else:
                sound3 = sfx3*amp3
                sp3=0
                ep3=sig_length
            if len(sfx4)<sig_length:
                sound4,sp4,ep4 = single_sound(sfx4,amp4,sig_length)
            else:
                sound4 = sfx4*amp4
                sp4=0
                ep4=sig_length
                
            mix = spk+sound1+sound2+sound3+sound4
            
            #savename = f'spk1_sfx3_{i}_13'  #
            savename = f'{title}_{name}_{l1}{l2}{l3}{l4}x_{sp_num}_{so_num}'
            print(savename)
            seffect = f'{seffect1} // {seffect2} // {seffect3} // {seffect4}'
            detail = f'{sp1} {ep1} {amp1} // {sp2} {ep2} {amp2} // {sp3} {ep3} {amp3} // {sp4} {ep4} {amp4}'
            usedfiles.append([speech,seffect,str(sp_num), str(so_num),detail]) #'1','3'])  #speaker file, sound effect file, num of speakers, num of sounds
            sf.write(folder+savename+'.wav',mix,fs,'PCM_16')
            if j==len(s4)-1:  #added this for permutations of sfx types. Apr 28
                j=0
            else:
                j=j+1
                    
        if splitter*4-1<i<splitter*5: #<i<1875:  #sfx 5
            #spk,fs = librosa.load(speech,sr=16000)  #load speech file
            so_num = 5
            
            ind_type = random.sample([0,1,2,3,4],so_num) #sfx type selection
            ind_file = random.sample(range(length_sfx),so_num) #sfx file selections
            
            seffect1 = sfxtrain[ind_type[0]][ind_file[0]]  #get filenames
            seffect2 = sfxtrain[ind_type[1]][ind_file[1]]
            seffect3 = sfxtrain[ind_type[2]][ind_file[2]]
            seffect4 = sfxtrain[ind_type[3]][ind_file[3]]
            seffect5 = sfxtrain[ind_type[4]][ind_file[4]]
            
            amp1,_ = amplitude_set(seffect1)
            amp2,_ = amplitude_set(seffect2)
            amp3,_ = amplitude_set(seffect3)
            amp4,_ = amplitude_set(seffect4)
            amp5,_ = amplitude_set(seffect5)
            
            sfx1,sam = librosa.load(seffect1,sr=16000)
            sfx2,sam = librosa.load(seffect2,sr=16000)
            sfx3,sam = librosa.load(seffect3,sr=16000)
            sfx4,sam = librosa.load(seffect4,sr=16000)
            sfx5,sam = librosa.load(seffect5,sr=16000)
            
            if len(sfx1)<sig_length:
                sound1,sp1,ep1 = single_sound(sfx1,amp1,sig_length)
            else:
                sound1 = sfx1*amp1  #outdoor sound effect
                sp1=0
                ep1=sig_length
                
            if len(sfx2)<sig_length:
                sound2,sp2,ep2 = single_sound(sfx2,amp2,sig_length)
            else:
                sound2 = sfx2*amp2
                sp2=0
                ep2=sig_length
                
            if len(sfx3)<sig_length:
                sound3,sp3,ep3 = single_sound(sfx3,amp3,sig_length)
            else:
                sound3 = sfx3*amp3
                sp3=0
                ep3=sig_length
                
            if len(sfx4)<sig_length:
                sound4,sp4,ep4 = single_sound(sfx4,amp4,sig_length)
            else:
                sound4 = sfx4*amp4
                sp4=0
                ep4=sig_length
                
            if len(sfx5)<sig_length:
                sound5,sp5,ep5 = single_sound(sfx5,amp5,sig_length)
            else:
                sound5 = sfx5*amp5
                sp5=0
                ep5=sig_length
                
            mix = spk+sound1+sound2+sound3+sound4+sound5
            
            #savename = f'spk1_sfx3_{i}_13'  #
            savename = f'{title}_{name}_dpkno_{sp_num}_{so_num}'
            print(savename)
            seffect = f'{seffect1} // {seffect2} // {seffect3} // {seffect4} // {seffect5}'
            detail = f'{sp1} {ep1} {amp1} // {sp2} {ep2} {amp2} // {sp3} {ep3} {amp3} // {sp4} {ep4} {amp4} // {sp5} {ep5} {amp5}'
            usedfiles.append([speech,seffect,str(sp_num), str(so_num),detail]) #'1','3'])  #speaker file, sound effect file, num of speakers, num of sounds
            sf.write(folder+savename+'.wav',mix,fs,'PCM_16')
            
        if i>splitter*5-1: #i>1874:  #no sound effect
            mix,fs = librosa.load(speech,sr=16000)
            so_num = 0
            usedfiles.append([speech,'none',str(sp_num),'0','none'])
            savename = f'{title}_{name}_xxxxx_{sp_num}_{so_num}'
            print(savename)
            sf.write(folder+savename+'.wav',mix,fs,'PCM_16')
            
    return usedfiles
#%
def mix_speech_test(speechfiles,sfxtrain,sp_num,total,folder):  
    #1. pass glob for 1 spk, then 2 spk, 3, 4 test file. Use 750 files in training per speaker class
    #2 : the master sound effect training file list of lists
    #3: number of speakers 1,2,3 or 4
    #4. number of samples
    #5: folder to save ie. 'SARfull/'
    import math
    from itertools import combinations
    #total = 546 #len(speechfiles)
    splitter = math.floor(total/6)
    
    flat_sfxtrain = [item for sublist in sfxtrain for item in sublist]  #flatten. 75 dimensional
    tok=0
    usedfiles=[]
    
    #sig_length = 160000
    for i, speech in enumerate(speechfiles):
        
        if i>total-1: #504: #504 per speaker class. to make it round 84 per sound class #749:
            break
        spk,fs = librosa.load(speech,sr=16000)
        sig_length = len(spk)
        name = speech[19:-6]
        if i<splitter: #i<84:  #1 sound effect. 504/6 num sound class= 84 for each sound class
            so_num = 1
            seffect = flat_sfxtrain[tok]  #sound effect file name
            #spk,fs = librosa.load(speech,sr=16000)
            sfx,fs1=librosa.load(seffect,sr=16000)
            amp,l1 = amplitude_set(seffect)
                
            if len(sfx)<sig_length:
                sound,sp1,ep1 = single_sound(sfx,amp,sig_length)
            else:
                sound = sfx*amp  #outdoor sound effect
                sp1=0
                ep1=sig_length
                
            mix = spk+sound
            savename = f'test_{name}_{l1}xxxx_{sp_num}_{so_num}'
            detail = f'{sp1} {ep1} {amp}'
            usedfiles.append([speech,seffect,str(sp_num), str(so_num),detail])#'1','1'])  #speaker file, sound effect file, num of speakers
            sf.write(folder+savename+'.wav',mix,fs,'PCM_16') 
            tok=tok+1
            if tok==len(flat_sfxtrain):#25:  #75 sound effect files in training split. change to 25 for test set
                tok=0  #start list from beginning. should result in equal representation of sound effect types 
        j=0
        if splitter-1<i<splitter*2: #83<i<168:  #2 random sounds. repeat process for 1 sound effect
            #spk,fs = librosa.load(speech,sr=16000)  #load speech file
            so_num = 2
            s2 = list(combinations([0,1,2,3,4],so_num))
            random.shuffle(s2)
            #ind_type = random.sample([0,1,2,3,4],so_num) #sfx type selection
            ind_file = random.sample(range(5),so_num) #sfx file selections
            
            seffect1 = sfxtrain[s2[j][0]][ind_file[0]]  #get filenames
            seffect2 = sfxtrain[s2[j][1]][ind_file[1]]
            
            amp1,l1 = amplitude_set(seffect1)
            amp2,l2 = amplitude_set(seffect2)
            
            sfx1,sam = librosa.load(seffect1,sr=16000)
            sfx2,sam = librosa.load(seffect2,sr=16000)
            
            if len(sfx1)<sig_length:
                sound1,sp1,ep1 = single_sound(sfx1,amp1,sig_length)
            else:
                sound1 = sfx1*amp1  #outdoor sound effect
                sp1=0
                ep1=sig_length
            
            if len(sfx2)<sig_length:
                sound2,sp2,ep2 = single_sound(sfx2,amp2,sig_length)
            else:
                sound2 = sfx2*amp2
                sp2=0
                ep2=sig_length
                
            mix = spk+sound1+sound2
            
            #savename = f'spk1_sfx2_{i}_12'  #
            savename = f'test_{name}_{l1}{l2}xxx_{sp_num}_{so_num}'
            seffect = f'{seffect1} // {seffect2}'
            detail = f'{sp1} {ep1} {amp1} // {sp2} {ep2} {amp2}'
            usedfiles.append([speech,seffect,str(sp_num), str(so_num),detail])#'1','2'])  #speaker file, sound effect file, num of speakers, num of sounds
            sf.write(folder+savename+'.wav',mix,fs,'PCM_16') 
            if j==len(s2)-1:
                j=0
            else:
                j=j+1
        j=0
        if splitter*2-1<i<splitter*3: #<i<252:  #sfx 3
            #spk,fs = librosa.load(speech,sr=16000)  #load speech file
            so_num = 3
            s3 = list(combinations([0,1,2,3,4],so_num))
            random.shuffle(s3)
            #ind_type = random.sample([0,1,2,3,4],so_num) #sfx type selection
            ind_file = random.sample(range(5),so_num) #sfx file selections
            
            seffect1 = sfxtrain[s3[j][0]][ind_file[0]]  #get filenames
            seffect2 = sfxtrain[s3[j][1]][ind_file[1]]
            seffect3 = sfxtrain[s3[j][2]][ind_file[2]]
            
            amp1,l1 = amplitude_set(seffect1)
            amp2,l2 = amplitude_set(seffect2)
            amp3,l3 = amplitude_set(seffect3)
            
            sfx1,sam = librosa.load(seffect1,sr=16000)
            sfx2,sam = librosa.load(seffect2,sr=16000)
            sfx3,sam = librosa.load(seffect3,sr=16000)
            
            if len(sfx1)<sig_length:
                sound1,sp1,ep1 = single_sound(sfx1,amp1,sig_length)
            else:
                sound1 = sfx1*amp1  #outdoor sound effect
                sp1=0
                ep1=sig_length
            
            if len(sfx2)<sig_length:
                sound2,sp2,ep2 = single_sound(sfx2,amp2,sig_length)
            else:
                sound2 = sfx2*amp2
                sp2=0
                ep2=sig_length
            if len(sfx3)<sig_length:
                sound3,sp3,ep3 = single_sound(sfx3,amp3,sig_length)
            else:
                sound3 = sfx3*amp3
                sp3=0
                ep3=sig_length
                
            mix = spk+sound1+sound2+sound3
            
            #savename = f'spk1_sfx3_{i}_13'  #
            savename = f'test_{name}_{l1}{l2}{l3}xx_{sp_num}_{so_num}'
            seffect = f'{seffect1} // {seffect2} // {seffect3}'
            detail = f'{sp1} {ep1} {amp1} // {sp2} {ep2} {amp2} // {sp3} {ep3} {amp3}'            
            usedfiles.append([speech,seffect,str(sp_num), str(so_num),detail]) #'1','3'])  #speaker file, sound effect file, num of speakers, num of sounds
            sf.write(folder+savename+'.wav',mix,fs,'PCM_16')
            if j==len(s3)-1:
                j=0
            else:
                j=j+1
                
        j=0            
        if splitter*3-1<i<splitter*4: #251<i<336:  #sfx 4
            #spk,fs = librosa.load(speech,sr=16000)  #load speech file
            so_num = 4
            s4 = list(combinations([0,1,2,3,4],so_num))
            random.shuffle(s4)
            ind_type = random.sample([0,1,2,3,4],so_num) #sfx type selection
            ind_file = random.sample(range(5),so_num) #sfx file selections
            
            seffect1 = sfxtrain[s4[j][0]][ind_file[0]]  #get filenames
            seffect2 = sfxtrain[s4[j][1]][ind_file[1]]
            seffect3 = sfxtrain[s4[j][2]][ind_file[2]]
            seffect4 = sfxtrain[s4[j][3]][ind_file[3]]
            
            amp1,l1 = amplitude_set(seffect1)
            amp2,l2 = amplitude_set(seffect2)
            amp3,l3 = amplitude_set(seffect3)
            amp4,l4 = amplitude_set(seffect4)
            
            sfx1,sam = librosa.load(seffect1,sr=16000)
            sfx2,sam = librosa.load(seffect2,sr=16000)
            sfx3,sam = librosa.load(seffect3,sr=16000)
            sfx4,sam = librosa.load(seffect4,sr=16000)
            
            if len(sfx1)<sig_length:
                sound1,sp1,ep1 = single_sound(sfx1,amp1,sig_length)
            else:
                sound1 = sfx1*amp1  #outdoor sound effect
                sp1=0
                ep1=sig_length
            
            if len(sfx2)<sig_length:
                sound2,sp2,ep2 = single_sound(sfx2,amp2,sig_length)
            else:
                sound2 = sfx2*amp2
                sp2=0
                ep2=sig_length
            if len(sfx3)<sig_length:
                sound3,sp3,ep3 = single_sound(sfx3,amp3,sig_length)
            else:
                sound3 = sfx3*amp3
                sp3=0
                ep3=sig_length
            if len(sfx4)<sig_length:
                sound4,sp4,ep4 = single_sound(sfx4,amp4,sig_length)
            else:
                sound4 = sfx4*amp4
                sp4=0
                ep4=sig_length
                
            mix = spk+sound1+sound2+sound3+sound4
            
            #savename = f'spk1_sfx3_{i}_13'  #
            savename = f'test_{name}_{l1}{l2}{l3}{l4}x_{sp_num}_{so_num}'
            seffect = f'{seffect1} // {seffect2} // {seffect3} // {seffect4}'
            detail = f'{sp1} {ep1} {amp1} // {sp2} {ep2} {amp2} // {sp3} {ep3} {amp3} // {sp4} {ep4} {amp4}'            
            usedfiles.append([speech,seffect,str(sp_num), str(so_num),detail]) #'1','3'])  #speaker file, sound effect file, num of speakers, num of sounds
            sf.write(folder+savename+'.wav',mix,fs,'PCM_16')
            if j==len(s4)-1:
                j=0
            else:
                j=j+1
            
        if splitter*4-1<i<splitter*5: #335<i<420:  #sfx 5
            #spk,fs = librosa.load(speech,sr=16000)  #load speech file
            so_num = 5
            
            ind_type = random.sample([0,1,2,3,4],so_num) #sfx type selection
            ind_file = random.sample(range(5),so_num) #sfx file selections
            
            seffect1 = sfxtrain[ind_type[0]][ind_file[0]]  #get filenames
            seffect2 = sfxtrain[ind_type[1]][ind_file[1]]
            seffect3 = sfxtrain[ind_type[2]][ind_file[2]]
            seffect4 = sfxtrain[ind_type[3]][ind_file[3]]
            seffect5 = sfxtrain[ind_type[4]][ind_file[4]]
            
            amp1,_ = amplitude_set(seffect1)
            amp2,_ = amplitude_set(seffect2)
            amp3,_ = amplitude_set(seffect3)
            amp4,_ = amplitude_set(seffect4)
            amp5,_ = amplitude_set(seffect5)
            
            sfx1,sam = librosa.load(seffect1,sr=16000)
            sfx2,sam = librosa.load(seffect2,sr=16000)
            sfx3,sam = librosa.load(seffect3,sr=16000)
            sfx4,sam = librosa.load(seffect4,sr=16000)
            sfx5,sam = librosa.load(seffect5,sr=16000)
            
            if len(sfx1)<sig_length:
                sound1,sp1,ep1 = single_sound(sfx1,amp1,sig_length)
            else:
                sound1 = sfx1*amp1  #outdoor sound effect
                sp1=0
                ep1=sig_length
                
            if len(sfx2)<sig_length:
                sound2,sp2,ep2 = single_sound(sfx2,amp2,sig_length)
            else:
                sound2 = sfx2*amp2
                sp2=0
                ep2=sig_length
                
            if len(sfx3)<sig_length:
                sound3,sp3,ep3 = single_sound(sfx3,amp3,sig_length)
            else:
                sound3 = sfx3*amp3
                sp3=0
                ep3=sig_length
                
            if len(sfx4)<sig_length:
                sound4,sp4,ep4 = single_sound(sfx4,amp4,sig_length)
            else:
                sound4 = sfx4*amp4
                sp4=0
                ep4=sig_length
                
            if len(sfx5)<sig_length:
                sound5,sp5,ep5 = single_sound(sfx5,amp5,sig_length)
            else:
                sound5 = sfx5*amp5
                sp5=0
                ep5=sig_length
                
            mix = spk+sound1+sound2+sound3+sound4+sound5
            
            #savename = f'spk1_sfx3_{i}_13'  #
            savename = f'test_{name}_dpkno_{sp_num}_{so_num}'
            seffect = f'{seffect1} // {seffect2} // {seffect3} // {seffect4} // {seffect5}'
            detail = f'{sp1} {ep1} {amp1} // {sp2} {ep2} {amp2} // {sp3} {ep3} {amp3} // {sp4} {ep4} {amp4} // {sp5} {ep5} {amp5}'
            usedfiles.append([speech,seffect,str(sp_num), str(so_num),detail]) #'1','3'])  #speaker file, sound effect file, num of speakers, num of sounds
            sf.write(folder+savename+'.wav',mix,fs,'PCM_16')
            
        if i>splitter*5-1: #i>419:  #no sound effect
            mix,fs = librosa.load(speech,sr=16000)
            so_num = 0
            usedfiles.append([speech,'none',str(sp_num),'0','none'])
            savename = f'test_{name}_xxxxx_{sp_num}_{so_num}'
            sf.write(folder+savename+'.wav',mix,fs,'PCM_16')
            
    return usedfiles  
#%% trying to split into train/val/test. AMI full-corpus partition
chunkset = 'AMI_chunks_v4/'
testy = ['TS3003','TS3007','ES2014','ES2004','EN2002','IS1009b','EN2002a','IS1009c','TS3003a','EN2002b','TS3003b','EN2002c',
        'TS3003c','EN2002d',
        'TS3003d','ES2004a','TS3007a','ES2004b','TS3007b','ES2004c','TS3007c','ES2004d',
        'TS3007d','ES2014a','ES2014b','ES2014c','ES2014d','IS1009a','IS1009d']
validy = ['ES2003a','ES2003b','ES2003c','ES2003d','ES2011a','ES2011b','ES2011c','ES2011d',
         'IB4001','IB4002','IB4003','IB4004','IB4010','IB4011',
         'IS1008a','IS1008b','IS1008c','IS1008d','TS3004a','TS3004b','TS3004c','TS3004d',
         'TS3006a','TS3006b','TS3006c','TS3006d']
trainy = ['ES2002', 'ES2005', 'ES2006', 'ES2007', 'ES2008', 'ES2009', 'ES2010',
          'ES2012', 'ES2013', 'ES2015', 'ES2016', 'IS1000', 'IS1001', 'IS1002', 'IS1003',
          'IS1004', 'IS1005', 'IS1006', 'IS1007', 'TS3005', 'TS3008', 'TS3009',
          'TS3010', 'TS3011', 'TS3012', 'EN2001', 'EN2003', 'EN2004a', 'EN2005a',
          'EN2006', 'EN2009', 'IN1001', 'IN1002', 'IN1005', 'IN1007', 'IN1008',
          'IN1009', 'IN1012', 'IN1013', 'IN1014', 'IN1016']
def filelist(validy,chunkset):
    validfiles=[[],[],[],[]]
    for t in validy:
        files = glob.glob(chunkset+t+'*.wav')  #get files to a meeting
        for f in files:
            if int(f[-5])==1:  #assigne meeting to correct list column by speaker #
                validfiles[0].append(f)
            if int(f[-5])==2:
                validfiles[1].append(f)
            if int(f[-5])==3:
                validfiles[2].append(f)
            if int(f[-5])==4:
                validfiles[3].append(f)
    return validfiles

trainfiles = filelist(trainy,chunkset)
validfiles = filelist(validy,chunkset)
#%% Train set creation
sound_train = [doors_tr, phone_tr, keyboard_tr, knocking_tr, outdoors_tr]
sound_train_flat = [item for sublist in sound_train for item in sublist]   

spe1 = glob.glob('AMI_chunks_v4/train/*_1.wav')
spe2 = glob.glob('AMI_chunks_v4/train/*_2.wav')
spe3 = glob.glob('AMI_chunks_v4/train/*_3.wav') 
spe4 = glob.glob('AMI_chunks_v4/train/*_4.wav') 
total = np.min([len(spe1),len(spe2),len(spe3),len(spe4)])
speaker1 = mix_speech_train(spe1,sound_train,1,total,'SAR_v4/')   #%%
speaker2 = mix_speech_train(spe2,sound_train,2,total,'SAR_v4/')
speaker3 = mix_speech_train(spe3,sound_train,3,total,'SAR_v4/')
speaker4 = mix_speech_train(spe4,sound_train,4,total,'SAR_v4/')

train = speaker1+speaker2+speaker3+speaker4
print('done train set.')
#%validation set
ech1 = glob.glob('AMI_chunks_v4/valid/*_1.wav')
ech2 = glob.glob('AMI_chunks_v4/valid/*_2.wav')
ech3 = glob.glob('AMI_chunks_v4/valid/*_3.wav') 
ech4 = glob.glob('AMI_chunks_v4/valid/*_4.wav')

total = np.min([len(ech1),len(ech2),len(ech3),len(ech4)])
sound_valid = [doors_v, phone_v, keyboard_v, knocking_v, outdoors_v]

valid1 = mix_speech_train(ech1,sound_valid,1,total,'SAR_v4/')   #%%
valid2 = mix_speech_train(ech2,sound_valid,2,total,'SAR_v4/')
valid3 = mix_speech_train(ech3,sound_valid,3,total,'SAR_v4/')
valid4 = mix_speech_train(ech4,sound_valid,4,total,'SAR_v4/')

valid = valid1+valid2+valid3+valid4

#%Test set creation
sound_test = [doors_te, phone_te, keyboard_te, knocking_te, outdoors_te]
speech1te = glob.glob('AMI_chunks_v4/test/*_1.wav')  
speech2te = glob.glob('AMI_chunks_v4/test/*_2.wav') 
speech3te = glob.glob('AMI_chunks_v4/test/*_3.wav')   
speech4te = glob.glob('AMI_chunks_v4/test/*_4.wav')   
total = np.min([len(speech1te),len(speech2te),len(speech3te),len(speech4te)])
#%
speak1te = mix_speech_test(speech1te,sound_test,1,total,'SAR_v4/')
speak2te = mix_speech_test(speech2te,sound_test,2,total,'SAR_v4/')
speak3te = mix_speech_test(speech3te,sound_test,3,total,'SAR_v4/')
speak4te = mix_speech_test(speech4te,sound_test,4,total,'SAR_v4/')

test = speak1te+speak2te+speak3te+speak4te

 
#%save list of dataset train and test mixing data as csv
#outdoor_singles(outdoors,o)  

import csv
with open('train_list_SAR_v4.csv','w') as f: #spk1_sfx1.csv','w') as f:
    writer = csv.writer(f, delimiter=',') #'\t')
    writer.writerows(train)
    
with open('test_list_SAR_v4.csv','w') as f: #spk1_sfx1.csv','w') as f:
    writer = csv.writer(f, delimiter=',') #'\t')
    writer.writerows(test)
    
with open('valid_list_SAR_v4.csv','w') as f: #spk1_sfx1.csv','w') as f:
    writer = csv.writer(f, delimiter=',') #'\t')
    writer.writerows(valid)
#%% Mixing speech and sound effect file testing code works
import librosa
import numpy as np

#speech_file = '/Users/Admin/Desktop/ami_ES2002a_test.wav'

speech_file = 'AMI_chunks_v4/train/EN2001a_0030_0040_2.wav' #'/Users/Admin/Downloads/Array1-01/ES2002a/audio/ES2002a.Array1-01.wav'
y1,fs1 = librosa.load(speech_file,sr=None) #,offset=70.0,duration=11.0)  #read at 1:10 mark for 11s

sound_file = '/Users/Admin/Downloads/dcase2016_task2_train_dev/dcase2016_task2_train_dev/dcase2016_task2_train/doorslam001.wav'
y2,fs2 = librosa.load(knocking_tr[0],sr=fs1)

temp = np.zeros(len(y1),dtype=np.float16)  #temporary matrix, size of speech signal
r1 = np.random.randint(len(y1)-len(y2))  #potential starting point range
r2 = r1+len(y2)  #endpoint of sound effect
temp[r1:r2] = y2 
A=1  #keep between 0.25 and0.5 for door; phone between 0.15 and 0.05; keyboard and knock 0.5 and 1
mixed = (y1+A*temp)

import matplotlib.pyplot as plt
import librosa.display
plt.subplot(1,3,1)
plt.plot(y1)
plt.title('speech')
plt.tight_layout()
plt.subplot(1,3,2)
plt.plot(y2)
plt.title('siren')
plt.tight_layout()
plt.subplot(1,3,3)
plt.plot(mixed)
plt.title('mix')
plt.tight_layout()
plt.show()
sf.write('mix.wav',mixed,16000,'PCM_16')
#%% Headset mixing
import soundfile as sf
import librosa
import numpy as np

a,f0 = librosa.load('/Users/Admin/Downloads/headset/ES2002d/audio/ES2002d.Headset-0.wav',sr=None)
b,f1 = librosa.load('/Users/Admin/Downloads/headset/ES2002d/audio/ES2002d.Headset-1.wav',sr=None)
c,f2 = librosa.load('/Users/Admin/Downloads/headset/ES2002d/audio/ES2002d.Headset-2.wav',sr=None)
d,f3 = librosa.load('/Users/Admin/Downloads/headset/ES2002d/audio/ES2002d.Headset-3.wav',sr=None)

mix = (a+b+c+d)

#sf.write('/Users/Admin/Downloads/ES2006d_mixed.wav',mix,f1,'PCM_16')

#%% Read path for each meeting, mix correspongding channels together, write to file

p = '/Users/Admin/Downloads/headset/'

import os
import librosa
import soundfile as sf
import numpy as np

sd=[]
for subdir, dirs, files in os.walk(p):  #Gather the subdirectory paths containing audio data
    print(subdir)
    if 'audio' in subdir:
        sd.append(subdir)
#%% 
for i in sd:  #loop through subdirectories paths
    for sub, di, fi in os.walk(i):  #get the file names in a subdirectory
        '''           
        if len(fi)==4:  # 4 files equals 4 channels of speakers in meeting
            
            a,f0 = librosa.load(sub+'/'+fi[0],sr=None)
            b,f1 = librosa.load(sub+'/'+fi[1],sr=None)
            c,f2 = librosa.load(sub+'/'+fi[2],sr=None)
            d,f3 = librosa.load(sub+'/'+fi[3],sr=None)
            
            if a.shape==b.shape and a.shape==c.shape and a.shape==d.shape:
                mix=a+b+c+d
                sf.write('/Users/Admin/Downloads/AMI_headset/'+fi[0][0:7]+'.wav', mix,f0,'PCM_16')
            else:
                print(sub+'/'+fi[0])
        '''          
        if len(fi)==5:  #use second if statement instead to get this to work
            a,f0 = librosa.load(sub+'/'+fi[0],sr=None)
            b,f1 = librosa.load(sub+'/'+fi[1],sr=None)
            c,f2 = librosa.load(sub+'/'+fi[2],sr=None)
            d,f3 = librosa.load(sub+'/'+fi[3],sr=None)
            e,f4 = librosa.load(sub+'/'+fi[4],sr=None)
            if a.shape==b.shape and a.shape==c.shape and a.shape==d.shape and a.shape==e.shape:
                mix=a+b+c+d+e
                sf.write('/Users/Admin/Downloads/AMI_headset/'+files[0][0:7]+'.wav', mix,f0,'PCM_16')
            else:
                print(sub+'/'+fi[0])
#%% 5 speaker meetings
import glob
sub = '/Users/Admin/Downloads/headset'

files = ['EN2001a','EN2001d','EN2001e']

for meeting in files:
    print(meeting, 'meeting')
    fi = glob.glob(sub+'/'+meeting+'/audio/*.wav')
    a,f0 = librosa.load(fi[0],sr=None)
    b,f1 = librosa.load(fi[1],sr=None)
    c,f2 = librosa.load(fi[2],sr=None)
    d,f3 = librosa.load(fi[3],sr=None)
    e,f4 = librosa.load(fi[4],sr=None)
    mix=a+b+c+d+e
    sf.write('/Users/Admin/Downloads/AMI_headset/'+meeting+'.wav',mix,f0,'PCM_16')