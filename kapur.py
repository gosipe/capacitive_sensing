#!/usr/bin/env python
# -*- coding:utf-8 -*-

from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt







def kapur_thresh(lick_trace,max_val):
    lick_hist, _ = np.histogram(lick_trace, bins=range(max_val), density=True)
    click_hist = lick_hist.cumsum()
    click_hist_i = 1.0 - click_hist
    # Check for invalid operations
    click_hist[click_hist <= 0] = 1
    click_hist_i[click_hist_i <= 0] = 1

    c_entropy = (lick_hist * np.log(lick_hist + (lick_hist <= 0))).cumsum()
    b_entropy = -c_entropy / click_hist + np.log(click_hist)

    c_entropy_i = c_entropy[-1] - c_entropy
    f_entropy = -c_entropy_i / click_hist_i + np.log(click_hist_i)

    # Plot b_entropy
    # plt.figure(figsize=(8, 6))
    # plt.plot(f_entropy, color='blue', label='f_entropy')
    # plt.plot(b_entropy, color='green', label='b_entropy')
    # plt.xlabel('Threshold Index')
    # plt.ylabel('Entropy')
    # plt.figure(figsize=(8, 6))
    # plt.plot(b_entropy + f_entropy, color='purple', label='Sum of Entropies')
    # plt.xlabel('Threshold Index')
    # plt.ylabel('Sum of Entropies')
    # plt.title('Sum of Background and Foreground Entropies')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    return np.argmax(b_entropy + f_entropy)



def otsu_threshold(lick_trace,max_val):
    # Compute histogram of lick data
    histogram, bin_edges = np.histogram(lick_trace, bins=max_val, range=(0, max_val))
    # Normalize histogram to get probabilities
    probabilities = histogram / np.sum(histogram)
    # Initialize variables
    max_variance = 0
    optimal_threshold = 0
    # Try all threshold values
    for threshold in range(1, max_val):
        # Calculate weight of background and foreground
        weight_background = np.sum(probabilities[:threshold])
        weight_foreground = np.sum(probabilities[threshold:])
        # If zero, continue to next threshold
        if weight_background == 0 or weight_foreground == 0:
            continue
        # Calculate mean background and foreground
        mean_background = np.sum(np.arange(threshold) * probabilities[:threshold]) / weight_background
        mean_foreground = np.sum(np.arange(threshold, max_val) * probabilities[threshold:]) / weight_foreground
        # Calculate between-class variance
        between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        # Update max variance and optimal threshold
        if between_class_variance > max_variance:
            max_variance = between_class_variance
            optimal_threshold = threshold
    return optimal_threshold


df = pd.read_csv('lick.csv')
x = df.iloc[:, 0]
y1 = df.iloc[:, 1]
y2 = df.iloc[:, 2]
y3 = df.iloc[:, 3]
y4 = df.iloc[:, 4]

y1_max=np.max(y1)
y1_kapur=kapur_threshold(y1,y1_max)
y1_otsu=otsu_threshold(y1,y1_max)

y2_max=np.max(y2)
y2_kapur=kapur_threshold(y2,y2_max)
y2_otsu=otsu_threshold(y2,y2_max)

y3_max=np.max(y3)
y3_kapur=kapur_threshold(y3,y3_max)
y3_otsu=otsu_threshold(y3,y3_max)

y4_max=np.max(y4)
y4_kapur=kapur_threshold(y4,y4_max)
y4_otsu=otsu_threshold(y4,y4_max)

x_minutes = x / 60  # Convert seconds to minutes
plt.figure(figsize=(10, 8))

plt.subplot(4, 1, 1)
plt.plot(x_minutes, y1, color='purple')
#plt.scatter(x[y1_lick_idx], y1[y1_lick_idx], color='magenta', label='Lick Bouts')
plt.axhline(y=y1_kapur, color='black', linestyle='--', label='Kapur', linewidth=1)
plt.axhline(y=y1_otsu, color='red', linestyle='--', label='Otsu', linewidth=1)
plt.xlabel('Time (min)')
plt.ylabel('Change in Capacitance')
plt.title('2 Females (BM7)')
plt.ylim(0, 160)
plt.yticks(np.arange(0, 161, 40))
plt.xlim(x_minutes.min(), 60)
plt.legend(loc='upper right')
plt.tick_params(axis= 'y',direction='in')
plt.tight_layout()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.subplot(4, 1, 2)
plt.plot(x_minutes, y2, color='green')
#plt.scatter(x[y2_lick_idx], y2[y2_lick_idx], color='magenta', label='Lick Bouts')
plt.axhline(y=y2_kapur, color='black', linestyle='--', label='Kapur', linewidth=1)
plt.axhline(y=y2_otsu, color='red', linestyle='--', label='Otsu', linewidth=1)
plt.xlabel('Time (min)')
plt.ylabel('Change in Capacitance')
plt.title('2 Males (BM8)')
plt.ylim(0, 160)
plt.yticks(np.arange(0, 161, 40))
plt.xlim(x_minutes.min(), 60)
plt.legend(loc='upper right')
plt.tick_params(axis= 'y',direction='in')
plt.tight_layout()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.subplot(4, 1, 3)
plt.plot(x_minutes, y3, color='purple')
#plt.scatter(x[y3_lick_idx], y3[y3_lick_idx], color='magenta', label='Lick Bouts')
plt.axhline(y=y3_kapur, color='black', linestyle='--', label='Kapur', linewidth=1)
plt.axhline(y=y3_otsu, color='red', linestyle='--', label='Otsu', linewidth=1)
plt.xlabel('Time (min)')
plt.ylabel('Change in Capacitance')
plt.title('2 Females (BM9)')
plt.ylim(0, 160)
plt.yticks(np.arange(0, 161, 40))
plt.xlim(x_minutes.min(), 60)
plt.legend(loc='upper right')
plt.tick_params(axis= 'y',direction='in')
plt.tight_layout()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.subplot(4, 1, 4)
plt.plot(x_minutes, y4, color='green')
#plt.scatter(x[y4_lick_idx], y4[y4_lick_idx], color='magenta', label='Lick Bouts')
plt.axhline(y=y4_kapur, color='black', linestyle='--', label='Kapur', linewidth=1)
plt.axhline(y=y4_otsu, color='red', linestyle='--', label='Otsu', linewidth=1)
plt.xlabel('Time (min)')
plt.ylabel('Change in Capacitance')
plt.title('2 Males (BM10)')
plt.ylim(0, 160)
plt.yticks(np.arange(0, 161, 40))
plt.xlim(x_minutes.min(), 60)
plt.legend(loc='upper right')
plt.tick_params(axis= 'y',direction='in')
plt.tight_layout()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show(block=False)

y_test=y4
y_test_max=np.max(y_test)
y_test_kapur=kapur_threshold(y_test,y_test_max)
k_5=y_test_kapur+(y_test_kapur*0.05)
k_10=y_test_kapur+(y_test_kapur*0.10)
k_25=y_test_kapur+(y_test_kapur*0.25)
k_50=y_test_kapur+(y_test_kapur*0.50)
plt.figure(figsize=(8, 6))
plt.plot(x_minutes, y_test, color='gray')
plt.axhline(y=y_test_kapur, color='black', linestyle='--', label='Kapur', linewidth=1)
plt.axhline(y=k_5, color='red', linestyle='--', label='K5', linewidth=1)
plt.axhline(y=k_10, color='orange', linestyle='--', label='K10', linewidth=1)
plt.axhline(y=k_25, color='green', linestyle='--', label='K25', linewidth=1)
plt.axhline(y=k_50, color='blue', linestyle='--', label='K50', linewidth=1)
plt.xlabel('Time (min)')
plt.ylabel('Change in Capacitance')
plt.title('2 Females (BM7)')
plt.ylim(0, 160)
plt.yticks(np.arange(0, 161, 40))
plt.xlim(x_minutes.min(), 60)
plt.legend(loc='upper right')
plt.tick_params(axis= 'y',direction='in')
plt.tight_layout()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()