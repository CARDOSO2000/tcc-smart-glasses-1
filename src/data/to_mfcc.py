
import csv
import os
import shutil
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from PIL import Image
import skimage.io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def mel_feature_extractor(file_name):
    
    y, sr = librosa.load(file_name)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    # Convert to log scale (dB). We'll use the peak power as reference.
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    return log_S

def feature_extractor(file_name):
    y_audio, sr = librosa.load(file_name)
    
    hop_length = int(sr/100)
    n_fft = int(sr/40)
    # 13? arbitrary number?
    mfcc_features = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13, hop_length=hop_length, n_fft=n_fft)
    return mfcc_features
    # mfcc_scaled_features = np.mean(mfcc_features.T, axis=0)
    
    # return mfcc_scaled_features

def save_image(mfcc, save_path):
    im =Image.fromarray(mfcc)
    if im.mode != 'RGB':
        im = im.convert('RGB') 
    # im.show()
    im.save(save_path)

def save_image_fig(mfcc, save_path):
    # https://stackoverflow.com/questions/52432731/store-the-spectrogram-as-image-in-python
    # For plotting headlessly
    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    
    # ax = fig.add_subplot(111)
    # ax.set_axis_off()
    
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    p = librosa.display.specshow(mfcc, ax=ax, y_axis='log', x_axis='time')
    fig.savefig(save_path)

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def save_image2(mfcc, save_path):
    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mfcc, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255-img # invert. make black==more energy

    # save as PNG
    skimage.io.imsave(save_path, img)

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

        path = data_path + "audio/fold" + str(row[5])+ "/" + str(row[0])
        
        # mfcc_features = feature_extractor(path)
        mel_features = mel_feature_extractor(path)
        save_path = processed_path + 'spectrograms/' + row[7] + '/' + row[0].split('.')[0] + '.jpg'
        save_image_fig(mfcc=mel_features, save_path=save_path)
        
        print (f"[{count}/{total_files}] fold{str(row[5])} saved to {save_path}")