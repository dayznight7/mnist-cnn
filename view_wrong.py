import pickle
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


with open("wrong_img_label_predict.pkl", "rb") as f:
    wrong_img, wrong_label, wrong_predict = pickle.load(f)

rows = 5
cols = 16
tot = len(wrong_label)
fig, axs = plt.subplots(rows, cols, figsize=(16, 5))

for i in range(rows * cols):
    row = i // 16
    col = i % 16
    if i < tot:
        img = wrong_img[i].squeeze()
        label = wrong_label[i]
        predict = wrong_predict[i]
        axs[row, col].imshow(img, cmap='gray')
        axs[row, col].set_title(f"L{label}P{predict}")
    axs[row, col].axis('off')


plt.show()
