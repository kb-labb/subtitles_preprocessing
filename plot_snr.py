import numpy as np
import os
from os import listdir,makedirs
import pandas as pd
import argparse
from tqdm import tqdm
import pdb
import re
import logging
import time
import librosa
import swifter
import matplotlib.pyplot as plt    
import numpy.random as rnd
from matplotlib import pyplot

df = pd.read_csv('snr.csv')
df["kanal"]= df["audio"].str[11:-34]
dfp = df.pivot(columns='kanal', values='snr')
ax = dfp.plot(kind='hist', figsize=(11, 6))
ax.set_xlabel("signal to noise",fontsize=12)
plt.savefig('snr.png')