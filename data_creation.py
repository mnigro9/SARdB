#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 16:56:43 2020

@author: Admin
"""
import os
import librosa
import soundfile as sf
import numpy as np
import glob

#%% 1. Headset mixing code
#%testing headset mixing works

a,f0 = librosa.load('/Users/Admin/Downloads/headset/ES2002d/audio/ES2002d.Headset-0.wav',sr=None)
b,f1 = librosa.load('/Users/Admin/Downloads/headset/ES2002d/audio/ES2002d.Headset-1.wav',sr=None)
c,f2 = librosa.load('/Users/Admin/Downloads/headset/ES2002d/audio/ES2002d.Headset-2.wav',sr=None)
d,f3 = librosa.load('/Users/Admin/Downloads/headset/ES2002d/audio/ES2002d.Headset-3.wav',sr=None)

mix = (a+b+c+d)

#sf.write('/Users/Admin/Downloads/ES2006d_mixed.wav',mix,f1,'PCM_16')

#%Read path for each meeting, mix correspongding channels together, write to file

p = '/Users/Admin/Downloads/headset/'


sd=[]
for subdir, dirs, files in os.walk(p):  #Gather the subdirectory paths containing audio data
    print(subdir)
    if 'audio' in subdir:
        sd.append(subdir)
#
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
#%5 speaker meetings 
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
#%% Functions for dataset creation 
def xml_parse(name,num_chunks,inter):
    import glob
    import xml.etree.ElementTree as ET
    import numpy as np
    nonos = [',','.','?','!','uh','um','mm','ah','huh','hmm','uh-huh','mm-hmm','uh-uh','mm-mm'
             'Uh','Um','Mm','Ah','Huh','Hmm','Uh-huh','Uh-Huh','Mm-hmm','Mm-Hmm','Uh-Uh','Uh-uh','Mm-mm','Mm-Mm']
    filenames = sorted(glob.glob('/Users/Admin/Downloads/ami_manual_1.6.1/words/'+name+'*.xml'))

    speechstarts= []
    speechends = []
    totalwords = []
    for i,file in enumerate(filenames):
        tree = ET.parse(file)  #set path to xml file
        root = tree.getroot()
        starttime=[]
        endtime = []
        words=[]
        for child in range(len(root)):  #iterate over each line in xml file 'root'
            #print(child.tag, child.attrib)
            label = root[child].attrib  #gets 'labels' of each line. A line indicates a single word
            
            if len(label)==3:  # condition met for line containing text
                w = root[child].text
                if (type(w) is str) and (any(x in w for x in nonos)==False): #check its a word said and not punctuation    
                    words.append(root[child].text)
                    starttime.append(float(label['starttime']))  #get numeric value of start end times for a spoken word
                    endtime.append(float(label['endtime']))
        totalwords.append(words)        
        speechstarts.append(starttime)
        speechends.append(endtime)
    t=0
    counts = []
    while t<num_chunks*inter: #10:  #should set to length of audio file
        tn = t+inter #10
        f0=0

        if len(speechstarts)==4:
            if any(t<x<tn for x in speechstarts[0]) and any(t<x<tn for x in speechends[0]):
                f0=f0+1
            if any(t<x<tn for x in speechstarts[1]) and any(t<x<tn for x in speechends[1]):
                f0=f0+1
            if any(t<x<tn for x in speechstarts[2]) and any(t<x<tn for x in speechends[2]): #checks if any value is in time interval
                f0=f0+1
            if any(t<x<tn for x in speechstarts[3]) and any(t<x<tn for x in speechends[3]): #checks if any value is in time interval
                f0=f0+1

        if len(speechstarts)==5:
            if any(t<x<tn for x in speechstarts[0]) and any(t<x<tn for x in speechends[0]):
                f0=f0+1
            if any(t<x<tn for x in speechstarts[1]) and any(t<x<tn for x in speechends[1]):
                f0=f0+1
            if any(t<x<tn for x in speechstarts[2]) and any(t<x<tn for x in speechends[2]): #checks if any value is in time interval
                f0=f0+1
            if any(t<x<tn for x in speechstarts[3]) and any(t<x<tn for x in speechends[3]): #checks if any value is in time interval
                f0=f0+1
            if any(t<x<tn for x in speechstarts[4]) and any(t<x<tn for x in speechends[4]):
                f0=f0+1
           
        counts.append(f0)        
        t=tn

    scriptfull=[]  #list containing a transcript for each segment of audio
    #whospokefull = [] #list of speaker id for each word
    if len(speechstarts)==4:
        for i in range(num_chunks):  #iterate through each segment
            t = i*inter
            tn = i*inter+inter
            s0 = [x for x in speechstarts[0] if t<x<tn ]  #number of words per speaker in segment timeinterval
            s1 = [x for x in speechstarts[1] if t<x<tn ]
            s2 = [x for x in speechstarts[2] if t<x<tn ]
            s3 = [x for x in speechstarts[3] if t<x<tn ]
            length_script = len(s0)+len(s1)+len(s2)+len(s3)  #number of words in segment interval
            script=[]
            #whospoke=[]
            for i in range(length_script):
                times = np.array([speechstarts[0][0], speechstarts[1][0], speechstarts[2][0], speechstarts[3][0]])  #current start time for each speaker
                current = np.argmin(times)  #position of word
                if speechends[current][0]<tn: #if endtime of word in interval
                    script.append([totalwords[current][0],current])
                    #whospoke.append(current)
                #script.append(totalwords[current][0])
                totalwords[current].pop(0)
                speechstarts[current].pop(0)
                speechends[current].pop(0)
                if len(speechstarts[current])==0:
                    speechstarts[current].append(1000000)
                    speechends[current].append(1000010)
            scriptfull.append(script)
            #whospokefull.append(whospoke)

    if len(speechstarts)==5:
        for i in range(num_chunks):  #iterate through each segment
            t = i*inter
            tn = i*inter+inter
            s0 = [x for x in speechstarts[0] if t<x<tn ]  #number of words per speaker in segment timeinterval
            s1 = [x for x in speechstarts[1] if t<x<tn ]
            s2 = [x for x in speechstarts[2] if t<x<tn ]
            s3 = [x for x in speechstarts[3] if t<x<tn ]
            s4 = [x for x in speechstarts[4] if t<x<tn ]
            length_script = len(s0)+len(s1)+len(s2)+len(s3)+len(s4)  #number of words in segment interval
            script=[]
            #whospoke=[]
            for i in range(length_script):
                times = np.array([speechstarts[0][0], speechstarts[1][0], speechstarts[2][0], speechstarts[3][0],speechstarts[4][0]])  #current start time for each speaker
                current = np.argmin(times)  #position of word
                if speechends[current][0]<tn: #if endtime of word in interval
                    script.append([totalwords[current][0],current])
                    #whospoke.append(current)
                #script.append(totalwords[current][0])
                totalwords[current].pop(0)
                speechstarts[current].pop(0)
                speechends[current].pop(0)
                if len(speechstarts[current])==0:
                    speechstarts[current].append(1000000)
                    speechends[current].append(1000010)
            scriptfull.append(script)
            #whospokefull.append(whospoke)

    return counts,scriptfull
# generate script of full meeting in order spoken            
    '''if len(speechstarts)==4:            
        length_script = len(totalwords[0])+len(totalwords[1])+len(totalwords[2])+len(totalwords[3])
        script=[]
        for i in range(length_script):
            times = np.array([speechstarts[0][0], speechstarts[1][0], speechstarts[2][0], speechstarts[3][0]])  #current start time for each speaker
            current = np.argmin(times)  #position of word
    
            script.append(totalwords[current][0])
            totalwords[current].pop(0)
            speechstarts[current].pop(0)
            if len(speechstarts[current])==0:
                speechstarts[current].append(10000) 
                
    if len(speechstarts)==5:
        length_script = len(totalwords[0])+len(totalwords[1])+len(totalwords[2])+len(totalwords[3])+len(totalwords[4])
        script=[]
        for i in range(length_script):
            times = np.array([speechstarts[0][0], speechstarts[1][0], speechstarts[2][0], speechstarts[3][0],speechstarts[4][0]])  #current start time for each speaker
            current = np.argmin(times)  #position of word
    
            script.append(totalwords[current][0])
            totalwords[current].pop(0)
            speechstarts[current].pop(0)
            if len(speechstarts[current])==0:
                speechstarts[current].append(1000000) 
        
    return counts,script'''

def scripture(words,ss):  ####NOT NEEDED JUST PROOF OF CONCEPT ######
    import numpy as np
    length_script = len(words[0])+len(words[1])+len(words[2])+len(words[3])
    script=[]
    for i in range(length_script):
        times = np.array([ss[0][0], ss[1][0], ss[2][0], ss[3][0]])  #current start time for each speaker
        current = np.argmin(times)  #position of word
    
        script.append(words[current][0])
        words[current].pop(0)
        ss[current].pop(0)
        if len(ss[current])==0:
            ss[current].append(10000)
    return script
#%% 2. 10s speech segment generated
'''
Going through each AMI file, chunk into 10s segments, save chunks with number 
of speakers.
'''
from pydub import AudioSegment
from pydub.utils import make_chunks

testy = ['IS1009b','EN2002a','IS1009c','TS3003a','EN2002b','TS3003b','EN2002c',
        'TS3003c','EN2002d',
        'TS3003d','ES2004a','TS3007a','ES2004b','TS3007b','ES2004c','TS3007c','ES2004d',
        'TS3007d','ES2014a','ES2014b','ES2014c','ES2014d','IS1009a','IS1009d']
# AMI files for non-scenario meetings listed here (EN,IB,IN)
ami_files = ['EN2001a','EN2001b','EN2001d','EN2001e',
             'EN2002a','EN2002b','EN2002d',  #EN2002c missing words doc D
             'EN2004a','EN2005a','EN2006a','EN2006b',  #EN2003a missing word doc D
             'EN2009d', #EN2009bc missing word doc D
             'ES2002a','ES2002c','ES2002d',
             'ES2003a','ES2003b','ES2003c','ES2003d',
             'ES2004a','ES2004b','ES2004c','ES2004d',
             'ES2005a','ES2005b','ES2005c','ES2005d',
             'ES2006a','ES2006b','ES2006c',
             'ES2007a','ES2007b','ES2007c','ES2007d',
             'ES2008a','ES2008b','ES2008d',
             'ES2009a','ES2009b','ES2009c','ES2009d',
             'ES2010a','ES2010b','ES2010c','ES2010d',
             'ES2011a','ES2011b','ES2011c','ES2011d',
             'ES2012a','ES2012b','ES2012c','ES2012d',
             'ES2013a','ES2013b','ES2013c','ES2013d',
             'ES2014a','ES2014b','ES2014c','ES2014d',
             'ES2015a','ES2015b','ES2015c','ES2015d',
             'ES2016a','ES2016b','ES2016c','ES2016d',
             'IB4001','IB4002','IB4003','IB4004','IB4005','IB4010','IB4011',
             'IN1002','IN1005','IN1007','IN1008','IN1009','IN1012','IN1013','IN1014','IN1016', #in1001 missind word doc
             'IS1000a','IS1000b','IS1000c','IS1000d',
             'IS1001a','IS1001b','IS1001c','IS1001d',
             'IS1002b','IS1002c','IS1002d',
             'IS1003a','IS1003b','IS1003c','IS1003d',
             'IS1004a','IS1004b','IS1004c','IS1004d',
             'IS1005a','IS1005b','IS1005c',
             'IS1006a','IS1006b','IS1006c','IS1006d',
             'IS1007a','IS1007b','IS1007c','IS1007d',
             'IS1008a','IS1008b','IS1008c','IS1008d',
             'IS1009a','IS1009b','IS1009c','IS1009d',
             'TS3003a','TS3003b','TS3003c','TS3003d',
             'TS3004a','TS3004b','TS3004c','TS3004d',
             'TS3005a','TS3005b','TS3005c','TS3005d',
             'TS3006a','TS3006b','TS3006c','TS3006d',
             'TS3007a','TS3007b','TS3007c','TS3007d',
             'TS3008a','TS3008b','TS3008c','TS3008d',
             'TS3009a','TS3009b','TS3009c','TS3009d',
             'TS3010a','TS3010b','TS3010c','TS3010d',
             'TS3011a','TS3011b','TS3011c','TS3011d',
             'TS3012a','TS3012b','TS3012c','TS3012d'
             ]

whereisaudio = '/Users/Admin/Documents/Spyder/AMI_headset/'
savehere = '/Users/Admin/Documents/Spyder/AMI_chunks_v6/'  #set correct folder directory
chunk_length_ms = 60000 # value is in miliseconds. Rounds up when making chunks if not divisible
cs = int(chunk_length_ms/1000) #chunk length in seconds

for file in ami_files:  
    myaudio = AudioSegment.from_file(whereisaudio+file+'.wav',"wav")  #load one file
    chunks = make_chunks(myaudio, chunk_length_ms) #makes 10s chunks as AudioSegment objects
    print(file)
    num_speakers,script  = xml_parse(file,len(chunks),cs)  #get number of speakers in each chunk
    for i, chunk in enumerate(chunks):
        start = cs*i  #eg. 0,10,20,30....
        stop = cs+cs*i #eg. 10,20,30,40....
        nspk = num_speakers[i]  #number of speakers in that chunk according to words.xml file 
        if nspk!=0:  #dont save 0 speaker segments
            chunk_name = f"{file}_{start:04}_{stop:04}_{nspk}"
            chunk.export(savehere+chunk_name+'.wav', format="wav")  #naming convention: 'meeting id','start_time','stop_time',num of speakers)
            with open(savehere+f"{file}_{start:04}_{stop:04}_script.txt",'w') as f:  #save script
                #f.write('\n'.join(script[i]))
                for tup in script[i]:
                    f.write(' '.join(str(x) for x in tup) + '\n')
                '''for row in script:
                    f.writelines(' '.join(row))
                    f.write('\n')'''

#%% 2. Getting RTTM files for speech segments
#ns,words = xml_parse('EN2001a',290,10)                
#ns,words,ss,se = xml_parse('EN2001a',290,10)
import xml.etree.ElementTree as ET
tree = ET.parse('/Users/Admin/Downloads/ami_manual_1.6.1/words/ES2002c.A.words.xml')
root = tree.getroot()
lab = root[10].text
print(root[10].attrib['endtime'])

testy = ['IS1009b','EN2002a','IS1009c','TS3003a','EN2002b','TS3003b','EN2002c',
        'TS3003c','EN2002d',
        'TS3003d','ES2004a','TS3007a','ES2004b','TS3007b','ES2004c','TS3007c','ES2004d',
        'TS3007d','ES2014a','ES2014b','ES2014c','ES2014d','IS1009a','IS1009d']

validy = ['ES2003a','ES2003b','ES2003c','ES2003d','ES2011a','ES2011b','ES2011c','ES2011d',
         'IB4001','IB4002','IB4003','IB4004','IB4010','IB4011',
         'IS1008a','IS1008b','IS1008c','IS1008d','TS3004a','TS3004b','TS3004c','TS3004d',
         'TS3006a','TS3006b','TS3006c','TS3006d']

trainy = ['ES2002a','ES2002c','ES2002d',
          'ES2005a','ES2005b','ES2005c','ES2005d',
          'ES2006a', 'ES2006b','ES2006c',
          'ES2007a', 'ES2007b','ES2007c','ES2007d',
          'ES2008a','ES2008b','ES2008d',
          'ES2009a','ES2009b','ES2009c','ES2009d',
          'ES2010a','ES2010b','ES2010c','ES2010d',
          'ES2012a','ES2012b','ES2012c','ES2012d',
          'ES2013a','ES2013b','ES2013c','ES2013d',
          'ES2015a','ES2015b','ES2015c','ES2015d',
          'ES2016a','ES2016b','ES2016c','ES2016d',
          'IS1000a','IS1000b','IS1000c','IS1000d',
          'IS1001a','IS1001b','IS1001c','IS1001d',
          'IS1002b','IS1002c','IS1002d',
          'IS1003a','IS1003b','IS1003c','IS1003d',
          'IS1004a','IS1004b','IS1004c','IS1004d',
          'IS1005a','IS1005b','IS1005c',
          'IS1006a','IS1006b','IS1006c','IS1006d',
          'IS1007a','IS1007b','IS1007c','IS1007d',
          'TS3005a','TS3005b','TS3005c','TS3005d',
          'TS3008a','TS3008b','TS3008c','TS3008d',
          'TS3009a','TS3009b','TS3009c','TS3009d',
          'TS3010a','TS3010b','TS3010c','TS3010d',
          'TS3011a','TS3011b','TS3011c','TS3011d',
          'TS3012a','TS3012b','TS3012c','TS3012d',
          'EN2001a','EN2001b','EN2001d','EN2001e',
          'EN2004a', 'EN2005a','EN2006a','EN2006b','EN2009d',
          'IN1002', 'IN1005', 'IN1007', 'IN1008',
          'IN1009', 'IN1012', 'IN1013', 'IN1014', 'IN1016']

def xml_rttm(name,typ):
    import glob
    import xml.etree.ElementTree as ET
    import numpy as np
    import pandas as pd
    nonos = [',','.','?','!','uh','um','mm','ah','huh','hmm','uh-huh','mm-hmm','uh-uh','mm-mm'
             'Uh','Um','Mm','Ah','Huh','Hmm','Uh-huh','Uh-Huh','Mm-hmm','Mm-Hmm','Uh-Uh','Uh-uh','Mm-mm','Mm-Mm']
    filenames = sorted(glob.glob('/Users/Admin/Downloads/ami_manual_1.6.1/words/'+name+'*.xml'))

    speechstarts= []
    speechends = []
    totalwords = []
    d=[]
    for i,file in enumerate(filenames):
        tree = ET.parse(file)  #set path to xml file
        root = tree.getroot()
        starttime=[]
        endtime = []
        words=[]
        spk = file[-11] 
        for child in range(len(root)):  #iterate over each line in xml file 'root'
            #print(child.tag, child.attrib)
            label = root[child].attrib  #gets 'labels' of each line. A line indicates a single word
            
            if len(label)==3:  # condition met for line containing text
                w = root[child].text
                if (type(w) is str) and (any(x in w for x in nonos)==False): #check its a word said and not punctuation    
                    words.append(root[child].text)
                    starttime.append(float(label['starttime']))  #get numeric value of start end times for a spoken word
                    endtime.append(float(label['endtime']))
                    start = float(label['starttime'])
                    duration = round(float(label['endtime']) - float(label['starttime']),2)
                    st = int(np.floor(start/10)*10) #37.../10
                    en = int(st+10)
                    fname = f'{typ}_{name}_{st:04}_{en:04}'
                    d.append(['SPEAKER',fname,1,start,duration,'<NA>','<NA>',spk,'<NA>','<NA>'])
                    #d.append(f'SPEAKER {name} 1 {start} {duration} <NA> <NA> {spk} <NA> <NA>')
                    #with open('testAMI.rttm') as d:
                    #    d.write(f'SPEAKER {name} 1 {start} {duration} <NA> <NA> {spk} <NA> <NA>')
                df = pd.DataFrame(d,columns=['Type','ID','Channel ID','Turn Onset','Turn Duration','Ortho','Spk type','Spk name','Score','Lookahead'])
                df = df.sort_values(by='Turn Onset')
    return df
#%%
d1 = xml_rttm('EN2001a','train')
#%%
import pandas as pd

for i,f in enumerate(trainy):
    print(f)
    d = xml_rttm(f,'train') #returns sorted dataframe as RTTM format
    if i==0:
        df = d
    if i>0:
        df = pd.concat([df,d],ignore_index=True)


#%%
df.to_csv('trainAMI.rttm',sep=' ',index=False,header=False)
#df = df_rttm(testy)

#%%
for i,f in enumerate(validy):
    d = xml_rttm(f,'valid') #returns sorted dataframe as RTTM format
    if i==0:
        dftr = d
    if i>0:
        dftr = pd.concat([dftr,d],ignore_index=True)
#%%
dftr.to_csv('validAMI.rttm',sep=' ',index=False,header=False)

#%% VAD WebRTC ....not working properly
#https://github.com/wiseman/py-webrtcvad/blob/3b39545dbb026d998bf407f1cb86e0ed6192a5a6/example.py
#  ############
'''
Not working great. Can say if the first frame provies False then its not speech?
Set 30ms frame size using the original code from wiseman .
'''

# #############    
import webrtcvad
import librosa

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    import collections
    import sys

    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for fram in frames:
        is_speech = vad.is_speech(fram.bytes, sample_rate)

        sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((fram, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(fram)
            ring_buffer.append((fram, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (fram.timestamp + fram.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (fram.timestamp + fram.duration))
    sys.stdout.write('\n')
    #If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])
        
def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

from librosa.util import frame  
import numpy as np  
   
vad = webrtcvad.Vad()  #create voice detector object
vad.set_mode(0)  # set aggressiveness parameter. 3 the most aggressive, 0 least

# VAD only taks 16bit mono PCM audio, at 8000 16000 3200 or 48kHz 
#speech_file = 'AMI_chunks_v3/EN2001a_0010_0020_3.wav' #'/Users/Admin/Downloads/Array1-01/ES2002a/audio/ES2002a.Array1-01.wav'
speech_file = 'AMI_chunks_v6/test/TS3007c_1260_1320_1.wav'
#speech_file = '/Users/Admin/Downloads/dcase2016_task2_train_dev/dcase2016_task2_train/knock014.wav'
audio,sample_rate = librosa.load(speech_file,sr=16000)
#audio2 = np.zeros((160000,))
#audio = audio[2137*sample_rate:2140*sample_rate,] #50-60s
#audio = np.concatenate((audio1,audio2))
if audio.dtype.kind == 'i':
    print('type i')
    if audio.max() > 2**15 - 1 or audio.min() < -2**15:
        raise ValueError(
                'when data type is int, data must be -32768 < data < 32767.')
    audio = audio.astype('f')

elif audio.dtype.kind == 'f':
    print('type f')
    if np.abs(audio).max() >= 1:
        audio = audio / np.abs(audio).max() * 0.9
        #warnings.warn('input data was rescaled.')
    audio = (audio * 2**15).astype('f')
else:
    raise ValueError('data dtype must be int or float.')


frames = frame_generator(30, audio, sample_rate)  #first arg is frame duartion in 10,20,30 ms
frs = list(frames)
valist2 = [vad.is_speech(tmp.bytes, 16000) for tmp in frs]

# from diarizationfunction.py
fs_vad = 16000
hoplength = 15 #ms  must be 10,20 or 30. value of 5,10,15works as 0.5 of 10,20,30 
hop = fs_vad * hoplength // 1000 #samp/s * ms * 1s/1000ms 
framelen = audio.size // hop + 1  #audio = audio.astype('int16') if not working
padlen = framelen * hop - audio.size
paded = np.lib.pad(audio, (0, padlen), 'constant', constant_values=0)
framed = frame(paded, frame_length=hop, hop_length=hop).T    
 
valist = [vad.is_speech(tmp.tobytes(), fs_vad) for tmp in framed]  #original for custom webrtc in diarizationfunction.oy
def framer(framed):  #function to generate Frame class from already segmented signal nd arrary
    for i,f in enumerate(framed):
        yield Frame(f,i*10,0.02)
frames2 = framer(framed)
frs2 = list(frames2)

#valist = [vad.is_speech(tmp.bytes, fs_vad) for tmp in frs2]

hop_origin = sample_rate * hoplength // 1000
va_framed = np.zeros([len(valist), hop_origin])
va_framed[valist] = 1
va = va_framed.reshape(-1)[:audio.size] #0 unvoiced, 1 is voiced. length of audio signal
'''
segments = vad_collector(sample_rate, 30,300,vad,frs)
print(segments[0])
'''
'''for c,i in enumerate(frs):
    if vad.is_speech(i.bytes,sample_rate): #only show frames with speech
        print('Frame ',c,' Contains speech:', vad.is_speech(i.bytes, sample_rate))'''