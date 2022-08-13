
# https://medium.com/deepdeploy/transfer-learning-for-sound-classification-c9696c931f7d
import csv
import os
import shutil
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import time

metadata_path = 'data/raw/UrbanSound8K/metadata/UrbanSound8K.csv'
data_path = 'data/raw/UrbanSound8K/'
processed_path = 'data/processed/'
count=-1

print('start')
if os.path.exists(processed_path):
    shutil.rmtree(processed_path)
    print('removed folders')

with open(metadata_path) as csvfile:
    total_files = sum(1 for row in csvfile)
    
print(f'open {metadata_path}')
with open(metadata_path) as csvfile:
    # goes through file 2 times, so I can have this total value
    
    print(f'Total files: {total_files}')
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        count+=1
        #skip header
        if count == 0:
            continue
            
        # print(count) 
        #row 7: class
        if not os.path.exists(processed_path + 'spectrograms/' + row[7]):
            os.makedirs(processed_path + 'spectrograms/' + row[7])
        
        # row 5 fold, row 0 slice file name
        y, sr = librosa.load(data_path + "audio/fold" + str(row[5])+ "/" + str(row[0]))

        # n mels?
        # Let's make and display a mel-scaled power (energy-squared) spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

        # Convert to log scale (dB). We'll use the peak power as reference.
        log_S = librosa.amplitude_to_db(S, ref=np.max)

        # Make a new figure
        fig = plt.figure(figsize=(12,4))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # Display the spectrogram on a mel scale
        # sample rate and hop length parameters are used to render the time axis
        librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

        # Make the figure layout compact
        
        #plt.show()
        save_path = processed_path + 'spectrograms/' + row[7] + '/' + row[0] + '.png'
        plt.savefig(save_path)
        plt.close()
        
        print (f"[{count}/{total_files}] Fold{str(row[5])} saved to {save_path}")