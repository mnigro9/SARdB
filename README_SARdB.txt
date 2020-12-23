SARdB READ ME

1. transcript
>> .txt files for corresponding meeting segments    
>> train, valid, test subfolders
 

2. SAR_v4   #SARdB audio files
>> 'test_TS3003d_0030_0040_dpkxx_1_3.wav'    'split_AMI meeting_meeting starttime_meeting endtime_sound events_number of speakers_number of sound events.wav'
>> >> 'split: train, valid, test
>> >> sound events: d(door slam), p(phone ringing), k(keyboard typing), n(knocking), o(outdoors), x(no sound)

3. my_pyannote_configrttm_files # for using pyannote.audio module
# pyannote.audio relies on pyannote.database that has config file indicating file locations. file should be in directiory root of datasets
>> config.yml file for module parameters
>> database.yml indicates file locations for rttm and uem types
>> .rttm 
SPEAKER {uri} 1 {start} {duration} <NA> <NA> {identifier} <NA> <NA>
so uri is filename, start is start time of speech turn in seconds
duration is in seconds, identfier is unique speaker identifier
SPEAKER ES2002b.Mix-Headset 1 14.4270 1.3310 <NA> <NA> FEE005 <NA> <NA>
>> .uem
UEM: tells what parts are annotated, use to cover full length of file in seconds
EN2001a.Mix-Headset 1 0.000 5250.240063 
>> >> ' SAR' files for speech only, 'SAR2 speech and sound event', SFX for sound event only

4. 'list_SAR' .csv files for metadata listing source files from AMI corpus and DCASE used in mixing process 

5. data_creation.py # python script for getting AMI meeting segments data. processing AMI corpus into 10s segments and extracting transcript

6. mixing_audio.py #script for making SARdB

7.nlp_util.py #produces text embeddings using Word2Vec and Sent2Vec

8. speaker_features.py #audio features script

9. uem_rttm.py  #script for generating uem and rttm file types