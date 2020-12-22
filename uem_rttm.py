#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 12:44:19 2020

@author: Admin

File for getting UEM files of pyannote.database setup and RTTM files
"""

import numpy as np
import pandas as pd
import glob
import xml.etree.ElementTree as ET
tree = ET.parse('/Users/Admin/Downloads/ami_manual_1.6.1/segments/TS3007a.A.segments.xml')
root = tree.getroot()
lab = root.text
#%%
# Generate UEM files for 10s long signals of SAR_V4 dataset
dtrain = sorted(glob.glob('/Users/Admin/Documents/Spyder/SAR_v4/train*.wav'))
dvalid = sorted(glob.glob('/Users/Admin/Documents/Spyder/SAR_v4/valid*.wav'))
dtest = sorted(glob.glob('/Users/Admin/Documents/Spyder/SAR_v4/test*.wav'))

#%%
def get_uem(dtrain):
    d=[]
    for file in dtrain:
        # {uri} 1 {start} {end}   filename 1 0 10
        d.append([file[37:],1,0,10])  #only want filename not full path
    df = pd.DataFrame(data=d,columns = ['uri','s','start','end'])
    return df

dft = get_uem(dtrain)
dfv = get_uem(dvalid)
dfe = get_uem(dtest)

dft.to_csv('trainAMI.uem',sep=' ',index=False,header=False)
dfv.to_csv('validAMI.uem',sep=' ',index=False,header=False)
dfe.to_csv('testAMI.uem',sep=' ',index=False,header=False)

#%% RTTM
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

def xml_rttm(name,typ,dtest):  #name: ami meeting name   type:train/valid/test  dtest: glob of sar dataset train/valid/test files
    import glob
    import xml.etree.ElementTree as ET
    import numpy as np
    import pandas as pd
    nonos = [',','.','?','!','uh','um','mm','ah','huh','hmm','uh-huh','mm-hmm','uh-uh','mm-mm'
             'Uh','Um','Mm','Ah','Huh','Hmm','Uh-huh','Uh-Huh','Mm-hmm','Mm-Hmm','Uh-Uh','Uh-uh','Mm-mm','Mm-Mm']
    filenames = sorted(glob.glob('/Users/Admin/Downloads/ami_manual_1.6.1/words/'+name+'*.xml'))
    puns = [',','.','?','!']
    d=[]
    for i,file in enumerate(filenames):
        tree = ET.parse(file)  #set path to xml file
        root = tree.getroot()
        
        spk = file[-11] 
        for child in range(len(root)):  #iterate over each line in xml file 'root'
            #print(child.tag, child.attrib)
            label = root[child].attrib  #gets 'labels' of each line. A line indicates a single word
            
            if len(label)==3:  # condition met for line containing text
                w = root[child].text    
                   
                if (type(w) is str) and (any(x in w for x in nonos)==False): #check its a word said and not punctuation    
                    start = float(label['starttime'])
                    st = int(np.floor(start/10)*10) #37.../10
                    en = int(st+10)
                    fname = f'{typ}_{name}_{st:04}_{en:04}'
                    if any(fname in s for s in dtest):  #checks that the segment is used in SAR dataset
                        matching = [s for s in dtest if fname in s]
                        fname = matching[0][37:-4] #updates the name to the SAR dataset filename used 
                        duration = round(float(label['endtime']) - float(label['starttime']),4)
                        start_norm = round((start-st)/(en-st)*10,4)
                        if start_norm+duration > 10.0:
                            duration = round(10.0 - start_norm,4)
                        d.append(['SPEAKER',fname,1,start_norm,duration,'<NA>','<NA>',spk,'<NA>','<NA>'])
                    #d.append(f'SPEAKER {name} 1 {start} {duration} <NA> <NA> {spk} <NA> <NA>')
                    #with open('testAMI.rttm') as d:
                    #    d.write(f'SPEAKER {name} 1 {start} {duration} <NA> <NA> {spk} <NA> <NA>')
    '''total = len(d)
    print(total)
    for i in range(total-1):
        print(f'{i}....{d[i]}')
        if len(d)==1:
            break
        if d[i][3]+d[i][4] == d[i+1][3]:  #if prev annotation extends to the next start
            d[i+1][3] = d[i][3]  #replace cell with new start time from previos annotation
            d[i+1][4] = d[i][4]+d[i+1][4] #replace cell with new duration adding successive durations
            print(d[i+1])
            #d.append(['SPEAKER',d[i][1],1,d[i][3],d[i][4]+d[i+1][4],'<NA>','<NA>',d[i][7],'<NA>','<NA>'])
            d.pop(0) #remove first in list after merging'''
    df = pd.DataFrame(d,columns=['Type','ID','Channel ID','Turn Onset','Turn Duration','Ortho','Spk type','Spk name','Score','Lookahead'])
    df = df.sort_values(by=['ID','Turn Onset'])
    return df


#%%test_TS3007d_1860_1870_dpkno_4_5
d1 = xml_rttm('TS3007d','test',dtest)
#%%
import pandas as pd

for i,f in enumerate(trainy):
    print(f)
    d = xml_rttm(f,'train',dtrain) #returns sorted dataframe as RTTM format
    if i==0:
        dftrain = d
    if i>0:
        dftrain = pd.concat([dftrain,d],ignore_index=True)

for i,f in enumerate(validy):
    d = xml_rttm(f,'valid',dvalid) #returns sorted dataframe as RTTM format
    if i==0:
        dfvalid = d
    if i>0:
        dfvalid = pd.concat([dfvalid,d],ignore_index=True)
        
for i,f in enumerate(testy):
    d = xml_rttm(f,'test',dtest) #returns sorted dataframe as RTTM format
    if i==0:
        dftest = d
    if i>0:
        dftest = pd.concat([dftest,d],ignore_index=True)
#%%
dftrain = dftrain.sort_values(by=['ID','Turn Onset'])
dftrain.to_csv('trainSAR.rttm',sep=' ',index=False,header=False)

dftest = dftest.sort_values(by=['ID','Turn Onset'])

dftest.to_csv('testSAR.rttm',sep=' ',index=False,header=False)

dfvalid = dfvalid.sort_values(by=['ID','Turn Onset'])
dfvalid.to_csv('validSAR.rttm',sep=' ',index=False,header=False)

#%% Sound Events RTTM generation
import csv

with open('/Users/Admin/Documents/Spyder/valid_list_SAR_v4.csv', newline='') as f:  ##### SET THIS
    reader = csv.reader(f)
    data = list(reader)
#%  
from operator import itemgetter
data = sorted(data, key=itemgetter(0))
#%
labels = {'p':'phone','k':'typing','o':'outdoors','n':'knocking','d':'doorslam'}
#%
sr=16000
s=[]
#%
def get_event(stype):
    if 'door' in stype:
        spk = 'doorslam'
    elif 'phone' in stype:
        spk = 'phone'
    elif 'audio' in stype:
        spk = 'outdoors'
    elif 'knock' in stype:
        spk = 'knocking'
    elif 'keyboard' in stype:
        spk = 'typing'
    return spk

director = dvalid ####### SET THIS
for i in range(len(data)):
    fname = director[i][37:-4]
    x = int(data[i][3])  # number of sound events
    sfx = data[i][4].split('//')  #sound event intervals
    stype = data[i][1].split('//')
    for t in range(x):
        spk = get_event(stype[t]) #labels[director[i][-13+t]]
        temp = sfx[t].lstrip(' ').partition(' ')
        start_norm = round(float(temp[0])/sr,2) #starttime in seconds rounded to 2
        temp2 = float(temp[2].partition(' ')[0])
        end_norm = round(temp2/sr,2)
        duration = round(end_norm - start_norm,2)
        s.append(['SPEAKER',fname,1,start_norm,duration,'<NA>','<NA>',spk,'<NA>','<NA>'])
#%
sdf = pd.DataFrame(s,columns=['Type','ID','Channel ID','Turn Onset','Turn Duration','Ortho','Spk type','Spk name','Score','Lookahead'])
sdf = sdf.sort_values(by=['ID','Turn Onset'])
sdf.to_csv('validSFX.rttm',sep=' ',index=False,header=False)
#%%
testSAR = pd.concat([dftest,sdf],ignore_index=True)  ##### SET THIS
testSAR = testSAR.sort_values(by=['ID','Turn Onset'])
testSAR.to_csv('testSAR2.rttm',sep=' ',index=False,header=False)  #####SET THIS

#%% SED labels  ' onset offset labelid'

import csv
import glob
import numpy as np
from operator import itemgetter
import librosa

_nfft = 2048 #256 origiinally
_hop = 511 #128
_nb_mel = 64 #40

def extract_mbe(_y, _sr, _nfft, _hop, _nb_mel):
    spec, n_fft = librosa.core.spectrum._spectrogram(y=_y, n_fft=_nfft, hop_length=_hop, power=1)
    mel_basis = librosa.filters.mel(sr=_sr, n_fft=_nfft, n_mels=_nb_mel)
    return np.log(np.dot(mel_basis, spec))

def get_event(stype):
    if 'door' in stype:
        spk = 'doorslam'
    elif 'phone' in stype:
        spk = 'phone'
    elif 'audio' in stype:
        spk = 'outdoors'
    elif 'knock' in stype:
        spk = 'knocking'
    elif 'keyboard' in stype:
        spk = 'typing'
    return spk

for split in ['train','valid','test']:

    with open(f'/Users/Admin/Documents/Spyder/{split}_list_SAR_v4.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
#%  
    data = sorted(data, key=itemgetter(0)) #sort by filename

    sr=16000
    sed_label,mbefull=[],[]
    savehere = 'SAR_v4_SED/'
    director = sorted(glob.glob(f'/Users/Admin/Documents/Spyder/SAR_v4/{split}*.wav')) 
    for i in range(len(data)):
        y,sr = librosa.load(director[i],sr=16000)
        mbe = extract_mbe(y,sr,_nfft,_hop,_nb_mel).T  #doing 256 FFT with 50% overlap and 40 mel bands
        fname = director[i][37:-4]
        x = int(data[i][3])  # number of sound events
        sfx = data[i][4].split('//')  #sound event intervals
        stype = data[i][1].split('//')
        s=[]
        for t in range(x):
            spk = get_event(stype[t]) # get the label id
            temp = sfx[t].lstrip(' ').partition(' ')
            start_norm = round(float(temp[0])/sr,2) #starttime in seconds rounded to 2
            temp2 = float(temp[2].partition(' ')[0])
            end_norm = round(temp2/sr,2)
            #duration = round(end_norm - start_norm,2)
            s.append([start_norm,end_norm,spk])
        s = sorted(s, key=itemgetter(0))  #sort by onset time (0) , sound event (2)
        '''with open(savehere+f"{fname}_SED.txt",'w') as f:  #save script
                wr = csv.writer(f,delimiter='\t') #'\n'.join(s))
                wr.writerows(s)'''
        lab = np.zeros((mbe.shape[0], 5)) #output of NN model shape
        for event in s:
            i1 = int(np.floor(event[0]*sr/128)) #divide by hop length for normalization 
            i2 = int(np.floor(event[1]*sr/128))
            if 'doorslam' in event[2]:
                lab[i1:i2,0] = 1
            if 'knocking' in event[2]:
                lab[i1:i2,1] = 1
            if 'outdoors' in event[2]:
                lab[i1:i2,2] = 1
            if 'phone' in event[2]:
                lab[i1:i2,3] = 1
            if 'typing' in event[2]:
                lab[i1:i2,4] = 1
        mbefull.append(mbe)
        sed_label.append(lab)
        
    np.savez(savehere+f"{split}_SEDv2",mbefull = mbefull, sed_label = sed_label)
'''
summation of each column of sed_label gives the number of active sources predicted. 
Loook into other ways of doing this. What should output layer activation be (sigmoid?softmax?)
'''
