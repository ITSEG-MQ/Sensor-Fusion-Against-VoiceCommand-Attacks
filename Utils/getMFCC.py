import os
import librosa
import numpy


BASE_DIR = "D:/"
# directory where your .wav files are
Directory_WAV = BASE_DIR + "Data_preprocessing/audio/"
# directory to put our results in
Directory_MFCC = BASE_DIR + "Data_preprocessing/stop_mfcc/"



# make a new folder in this directory to save our results in
if not os.path.exists(Directory_MFCC):
    os.makedirs(Directory_MFCC)

# get MFCCs for every .wav file in our specified directory
for filename in os.listdir(Directory_WAV):
    if filename.endswith('.wav'): # only get MFCCs from .wavs
        #print(filename)
        # read in our file
        y, sr = librosa.load(Directory_WAV + "/" +filename, sr=None)

        # get mfcc
        mfccs = librosa.feature.mfcc( y=y, sr=sr, n_mfcc=24)
        #print(mfccs.shape)

        # create a file to save our results in
        outputFile = Directory_MFCC + "/" + os.path.splitext(filename)[0] + ".mfcc"

        file = open(outputFile, 'w+') # make file/over write existing file
        numpy.savetxt(file, mfccs, delimiter=",") #save MFCCs as .mfcc
        file.close() # close file