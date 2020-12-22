#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 17:56:08 2020

@author: Admin

 utility for Gensim text vector modelling
"""
import glob
def read_ami(transcript):
    import csv
    punctuations='''[]{}!"'@#$%^&-_?.,'''  #chars to remove if seen in transcripts
    f=[]
    with open(transcript,'r') as s:
        reader2 = csv.reader(s, delimiter=' ')
        for row2 in reader2:
            f.append(row2[0])
        for char in range(len(f)):
            f[char] = f[char].lower()
    return f

tr = read_ami('AMI_chunks_v4/train/EN2004a_3150_3160_script.txt')

        
        
def find_nth(haystack, needle, n): #needed to get the corresponding script .txt name to audio signal of SARdb
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def wordvec_training(wtran,dict_size):
    '''
    # trained a Gensim Word2Vec model and produces Sentence2Vec model
    # arguments: training set text
                dictionary size
    '''
    from gensim.models import Word2Vec
    from fse.models import Sentence2Vec
    model = Word2Vec(wtran,min_count=0,size=dict_size)   #set size to time dimension of spectrograms
#above generates the word embeddings space model from my given data

    se = Sentence2Vec(model)
    return se

def get_sentvec(se,vtran):
    ''' function to generate Sentence2Vec embeddings from a list of transcripts
    '''
    vsent = se.train(vtran)
    return vsent

def get_embed(se,tmp):
    ''' for Sentence2Vec model se use infer on sentence tmp to get array of embedding vector '''
    a = se.infer([tmp])
    return a
#%% Start here. Get the scripts for SARdb train, valid, and test sets


    
setting = ['train','valid','test']
transcripts=[[],[],[]]
files=[[],[],[]]
nspks = [[],[],[]]
for s,_ in enumerate(setting):
    #rootnames = sorted(glob.glob(f'SAR_v4/{setting[0]}*.wav'))
    print('Starting ', setting[s])
    rootnames = sorted(glob.glob(f'SAR_v4/{setting[s]}*.wav'))
    for name in rootnames:
        start = name.find('_',5)  #starting point for file name
        end = find_nth(name,'_',5)
        files[s].append(f'AMI_chunks_v4/{setting[s]}/{name[start+1:end]}_script.txt')
        transcripts[s].append(read_ami(f'AMI_chunks_v4/{setting[s]}/{name[start+1:end]}_script.txt'))
        nspks[s].append(int(name[-7]))

#%%

se = wordvec_training(transcripts[0]+transcripts[1],200)

#%%
train_vec = get_sentvec(se,transcripts[0])
#%%
valid_vec = get_sentvec(se,transcripts[1])
test_vec = get_sentvec(se,transcripts[2])

#%%
import numpy as np
np.savez('SARdb_SentVec',train_vec=train_vec, valid_vec=valid_vec, test_vec=test_vec, files=files, nspks=nspks)