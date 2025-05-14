import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('lick.csv')

x = df.iloc[:, 0]
y1 = df.iloc[:, 1]
y2 = df.iloc[:, 2]
y3 = df.iloc[:, 3]
y4 = df.iloc[:, 4]

# Calculate percentiles and cutoffs
percentile_y1 = y1.quantile(0.99)
percentile_y2 = y2.quantile(0.99)
percentile_y3 = y3.quantile(0.999)
percentile_y4 = y4.quantile(0.99)

y1_cutoff = percentile_y1/2
y2_cutoff = percentile_y2/2
y3_cutoff = percentile_y3
y4_cutoff = percentile_y4/2

# Find lick bout indices
y1_lick_idx = y1 > y1_cutoff
y2_lick_idx = y2 > y2_cutoff
y3_lick_idx = y3 > y3_cutoff
y4_lick_idx = y4 > y4_cutoff

x_minutes = x / 60  # Convert seconds to minutes

plt.figure(figsize=(10, 8))

plt.subplot(4, 1, 1)
plt.plot(x_minutes, y1, color='purple')
#plt.scatter(x[y1_lick_idx], y1[y1_lick_idx], color='magenta', label='Lick Bouts')
plt.axhline(y=y1_cutoff, color='black', linestyle='--', label='Cutoff', linewidth=1)
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
plt.axhline(y=y2_cutoff, color='black', linestyle='--', label='Cutoff', linewidth=1)
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
plt.axhline(y=y3_cutoff, color='black', linestyle='--', label='Cutoff', linewidth=1)
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
plt.axhline(y=y4_cutoff, color='black', linestyle='--', label='Cutoff', linewidth=1)
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

#plt.show()

# Calculate the percentile for y1
percentile_y1 = y1.quantile(0.99)
#print(f"y1 (BM7): {percentile_y1}")

# Calculate the percentiles for y2, y3, and y4
percentile_y2 = y2.quantile(0.99)
#print(f"y2 (BM8): {percentile_y2}")

percentile_y3 = y3.quantile(0.999)
#print(f"y3 (BM9): {percentile_y3}")

percentile_y4 = y4.quantile(0.99)
#print(f"y4 (BM10): {percentile_y4}")

y1_cutoff = percentile_y1/2
y2_cutoff = percentile_y2/2
y3_cutoff = percentile_y3
y4_cutoff = percentile_y4/2

y1_lick_bouts = y1[y1 > y1_cutoff].values
y2_lick_bouts = y2[y2 > y2_cutoff].values
y3_lick_bouts = y3[y3 > y3_cutoff].values
y4_lick_bouts = y4[y4 > y4_cutoff].values

#print(f"Number of y1 lick bouts: {len(y1_lick_bouts)}")
#print(f"Number of y2 lick bouts: {len(y2_lick_bouts)}")
#print(f"Number of y3 lick bouts: {len(y3_lick_bouts)}")
#print(f"Number of y4 lick bouts: {len(y4_lick_bouts)}")

# Create a histogram of the values for y1
plt.figure(figsize=(8, 6))
plt.hist(y1, bins=30, color='blue', edgecolor='black')
plt.xlabel('Capacitance (BM7)')
plt.ylabel('Frequency')
plt.title('Histogram of BM7 Capacitance Values')
plt.tight_layout()
#plt.show()

# Create a histogram of the values for y2
plt.figure(figsize=(8, 6))
plt.hist(y2, bins=30, color='orange', edgecolor='black')
plt.xlabel('Capacitance (BM8)')
plt.ylabel('Frequency')
plt.title('Histogram of BM8 Capacitance Values')
plt.tight_layout()
#plt.show()

# Create a histogram of the values for y3
plt.figure(figsize=(8, 6))
plt.hist(y3, bins=30, color='green', edgecolor='black')
plt.xlabel('Capacitance (BM9)')
plt.ylabel('Frequency')
plt.title('Histogram of BM9 Capacitance Values')
plt.tight_layout()
#plt.show()

# Create a histogram of the values for y4
plt.figure(figsize=(8, 6))
plt.hist(y4, bins=30, color='red', edgecolor='black')
plt.xlabel('Capacitance (BM10)')
plt.ylabel('Frequency')
plt.title('Histogram of BM10 Capacitance Values')
plt.tight_layout()
#plt.show()

BM7_bottle_change = 2.86
BM8_bottle_change = 3.87
BM9_bottle_change = 0.05
BM10_bottle_change = 3.88

BM7_lick_bouts = len(y1_lick_bouts)
BM8_lick_bouts = len(y2_lick_bouts)
BM9_lick_bouts = len(y3_lick_bouts)
BM10_lick_bouts = len(y4_lick_bouts)

# Arrays for plotting
bottle_changes = np.array([BM7_bottle_change, BM8_bottle_change, BM9_bottle_change, BM10_bottle_change])
lick_bouts = np.array([BM7_lick_bouts, BM8_lick_bouts, BM9_lick_bouts, BM10_lick_bouts])

plt.figure(figsize=(7, 5))
plt.scatter(lick_bouts, bottle_changes, color='purple', label='Data Points')

# Line of best fit
m, b = np.polyfit(lick_bouts, bottle_changes, 1)
predicted = m * lick_bouts + b

# Manual R² calculation
ss_res = np.sum((bottle_changes - predicted) ** 2)
ss_tot = np.sum((bottle_changes - np.mean(bottle_changes)) ** 2)
r2 = 1 - (ss_res / ss_tot)

plt.plot(lick_bouts, predicted, color='black', linestyle='--', label='Best Fit Line')
plt.xlabel('Number of Lick Bouts')
plt.ylabel('Bottle Change')
plt.title(f'Bottle Change vs. Lick Bouts\n$R^2$ = {r2:.3f}')
plt.legend()
plt.tight_layout()
plt.show()