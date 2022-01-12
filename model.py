import pandas as pd
import numpy as np

def load_data(url):
    txt = requests.get(url).content
    txt = txt.decode('utf-8')
    x, y = [], []
    txt = txt.split('\n')
    for line in txt[:]:
        x_split = line.split(' ')
        for i in range(len(x_split)):
            x_split[i] = float(x_split[i])
        data.append(x_split)
    return data

