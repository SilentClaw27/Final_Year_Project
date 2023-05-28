#Packages imported for the functions used

import librosa
import librosa.display
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
import sys
import time
import shutil
from pathlib import Path
import utils
import pandas as pd

#Saving each respective mel-spectrogram into our directory
def save_melspectrogram(mel, path):
    librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()
    gc.collect(2)

gc.disable()
start = time.time()
counter = 0

main_folder = "fma_small"
dst_folder = "small"

#Going through each folder of the Dataset
for folder in os.listdir(main_folder):
    folder_path = os.path.join(main_folder, folder)
    if folder == "README.txt" or folder == "checksums":
        continue
    for file in os.listdir(folder_path):
        # Only process files
        if os.path.isfile(os.path.join(folder_path, file)):
            name = Path(file).stem
            # If the file has not been processed already
            if (name + '.png') in os.listdir(dst_folder):
                print("Skipped %d\r" % counter)
                sys.stdout.flush()
                counter += 1
                continue
            try:
                #Creating the melspectrogram and saving it into the required path 
                y, sr = librosa.load(os.path.join(folder_path, file))
                melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
                save_melspectrogram(melspectrogram, os.path.join(dst_folder, name + ".png"))
                print("%s\t%d\r" % (name, counter))
                sys.stdout.flush()
                counter += 1
                gc.collect(2)

            except Exception as e:
                print("failed...", e)
                counter += 1
                continue
end = time.time()