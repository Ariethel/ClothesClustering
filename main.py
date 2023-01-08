import matplotlib as matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from alive_progress import alive_bar
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import plotly.express as px
import os.path

def img_split(img, row, col):
    M = img.shape[0] // col  # larghezza singola colonna
    N = img.shape[1] // row  # altezza singola riga
    j = 0
    i = 0
    tiles = [[img[x:x + M, y:y + N] for x in range(0, img.shape[0] - (img.shape[0] % col), M) for y in
              range(0, img.shape[1] - (img.shape[1] % row), N)]]
    return tiles



def feature_extraction(df):
    # calcola le feature di ogni immagine
    with alive_bar(df.shape[0], force_tty = True) as bar:
        for idx, row in df.iterrows():
            img = cv2.imread(row['image_path'], 0)
            tiles = img_split(img, 3, 3)
            for index, tile in enumerate(tiles[0]):
                df.at[idx, f'non_empty_pixel_at_{index+1}'] = non_empty_pixel_feature(tile)
            bar()






def dataframe_clean(df):
    for col in df.columns:
        if col != 'category' and col != 'image_path':
            df.drop(col, axis=1, inplace=True)




def non_empty_pixel_feature(tile):
    # calcola il numero di pixel utilizzati
    return np.count_nonzero(tile)/tile.size


def plot_grid(grid, row, col, h=5, w=5):
    fig, ax = plt.subplots(nrows=row, ncols=col)
    [axi.set_axis_off() for axi in ax.ravel()]

    fig.set_figheight(h)
    fig.set_figwidth(w)
    c = 0
    for row in ax:
        for col in row:
            col.imshow(np.flip(grid[c], axis=-1))
            c += 1
    plt.show()


def main():
    img = cv2.imread("painting.png", 0)
    row, col = 3, 3
    tiles = img_split(img, row, col)
    if os.path.exists("csv/dataframe.csv"):
        df = pd.read_csv("csv/dataframe.csv")
    else:
        df = pd.read_csv("csv/train.csv")
        dataframe_clean(df)
        feature_extraction(df)
        df.to_csv("csv/dataframe.csv", index=False)
    model = KMeans(n_clusters=3)
    X = df[df.columns[2:]]
    y = df['category']
    #X = df.drop(['category', 'image_path'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train)
    predict = model.predict(X_test)


    X_test.at[:, 'label'] = predict
    X_test.at[:, 'category'] = y_test
    X_test = (X_test.groupby(['label', 'category']).size().reset_index(name='count'))
    #capire cosa rappresentano i cluster
    px.bar(X_test, x='category', y="count",  color="label", barmode="group").show()

if __name__ == '__main__':
    main()
