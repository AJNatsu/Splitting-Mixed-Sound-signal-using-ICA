import numpy as np
import scipy.signal as sp
import scipy as scipy
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import mean_squared_error

import soundfile as sf
x1_in, fs1 = sf.read('report2_wav 2/x1.wav')
x2_in, fs2 = sf.read('report2_wav 2/x2.wav')
x3_in, fs3 = sf.read('report2_wav 2/x3.wav')

s1_in, fs1 = sf.read('report2_wav 2/s1.wav')
s2_in, fs2 = sf.read('report2_wav 2/s2.wav')
s3_in, fs3 = sf.read('report2_wav 2/s3.wav')
# 標準化
M = len(x1_in)#信号のサンプル数
x = np.zeros((3, len(x1_in)))
x[0,:] = (x1_in - np.mean(x1_in))/np.sqrt(np.nanvar(x1_in))
x[1,:] = (x2_in - np.mean(x2_in))/np.sqrt(np.nanvar(x2_in))
x[2,:] = (x3_in - np.mean(x3_in))/np.sqrt(np.nanvar(x3_in))

# 観測信号の白色化化
R = np.dot(x, x.T)/M
eig_v, Q = np.linalg.eig(R)
Lambda = np.diag(eig_v)
V = np.dot(np.sqrt(np.linalg.inv(Lambda)), Q.T)
xh = np.dot(V,x)

b_past = np.eye(3)
b = np.eye(3)
# 1つ目の基底探索
for i in range(100):
  b_past[:, 0] = b[:, 0]
  sh1 = np.dot(b[:, 0].T, xh)
  b[0, 0] = np.mean(np.dot(np.power(sh1, 3), xh[0, :])) - 3*b[0, 0]
  b[1, 0] = np.mean(np.dot(np.power(sh1, 3), xh[1, :])) - 3*b[1, 0]
  b[2, 0] = np.mean(np.dot(np.power(sh1, 3), xh[2, :])) - 3*b[2, 0]
  b[:, 0] = b[:,0]/np.linalg.norm(b[:, 0])
  if np.abs(np.abs(np.dot(b[:, 0].T, b_past[:, 0]))) < 10**-6:
    break

# 2つ目の基底探索
for i in range(100):
  b_past[:, 1] = b[:, 1]
  # orthogonalize with respect to first basis vector
  b[:, 1] = b[:, 1] - np.dot(np.dot(b[:, 1].T, b[:, 0]), b[:, 0])
  sh2 = np.dot(b[:, 1].T, xh)
  # update the 2nd basis vector
  b[0, 1] = np.mean(np.dot(np.power(sh2, 3), xh[0, :])) - 3 * b[0, 1]
  b[1, 1] = np.mean(np.dot(np.power(sh2, 3), xh[1, :])) - 3 * b[1, 1]
  b[2, 1] = np.mean(np.dot(np.power(sh2, 3), xh[2, :])) - 3 * b[2, 1]
  # normalization
  b[:, 1] = b[:, 1] / np.linalg.norm(b[:, 1])
  if np.abs(np.abs(np.dot(b[:, 1].T, b_past[:, 1]))) < 10 ** -6:
    break

# 3つ目の基底探索
for i in range(100):
  b_past[:, 2] = b[:, 2]
  # orthogonalize with respect to first two basis vectors
  b[:, 2] = b[:, 2] - np.dot(np.dot(b[:, 2].T, b[:, 0]), b[:, 0]) - np.dot(np.dot(b[:, 2].T, b[:, 1]), b[:, 1])
  sh3 = np.dot(b[:, 2].T, xh)
  # update the 3rd basis vector
  b[0, 2] = np.mean(np.dot(np.power(sh3, 3), xh[0, :])) - 3 * b[0, 2]
  b[1, 2] = np.mean(np.dot(np.power(sh3, 3), xh[1, :])) - 3 * b[1, 2]
  b[2, 2] = np.mean(np.dot(np.power(sh3, 3), xh[2, :])) - 3 * b[2, 2]
  # normalization
  b[:, 2] = b[:, 2] / np.linalg.norm(b[:, 2])
  if np.abs(np.abs(np.dot(b[:, 2].T, b_past[:, 2]))) < 10 ** -6:
    break

# once have the basis vectors, separate the source signals
s = np.dot(b.T, xh)

# s[0, :], s[1, :], and s[2, :] approximately be the source signals s1, s2, and s3


sh1 = s[0, :]
sh2 = s[1, :]
sh3 = s[2, :]

# Compute the correlation matrix
corr_matrix = np.corrcoef(np.concatenate((np.array([s1_in, s2_in, s3_in]), np.array([sh1, sh2, sh3])), axis=0))

#power normalization
var_originals = np.var([s1_in, s2_in, s3_in], axis=1)
var_separated = np.var([sh1, sh2, sh3], axis=1)

sh1 *= np.sqrt(var_originals[0] / var_separated[0])
sh2 *= np.sqrt(var_originals[1] / var_separated[1])
sh3 *= np.sqrt(var_originals[2] / var_separated[2])

#phase correction
corr = np.corrcoef([s1_in, s2_in, s3_in, sh1, sh2, sh3])
corr_s_sh = corr[0:3, 3:6]  # Correlation between original and separated signals

for i in range(3):
    for j in range(3):
        if corr_s_sh[i, j] < 0:
            if i == 0:
                sh1 *= -1
            elif i == 1:
                sh2 *= -1
            else:
                sh3 *= -1

# Cross-correlation calculation
corr_s1_sh1 = np.correlate(s1_in, sh1, mode='valid')
corr_s1_sh2 = np.correlate(s1_in, sh2, mode='valid')
corr_s1_sh3 = np.correlate(s1_in, sh3, mode='valid')

corr_s2_sh1 = np.correlate(s2_in, sh1, mode='valid')
corr_s2_sh2 = np.correlate(s2_in, sh2, mode='valid')
corr_s2_sh3 = np.correlate(s2_in, sh3, mode='valid')

corr_s3_sh1 = np.correlate(s3_in, sh1, mode='valid')
corr_s3_sh2 = np.correlate(s3_in, sh2, mode='valid')
corr_s3_sh3 = np.correlate(s3_in, sh3, mode='valid')

# Creating a correlation matrix
cross_corr_matrix = np.array([[corr_s1_sh1, corr_s1_sh2, corr_s1_sh3],
                              [corr_s2_sh1, corr_s2_sh2, corr_s2_sh3],
                              [corr_s3_sh1, corr_s3_sh2, corr_s3_sh3]])

# Compute the maximum correlation for each separated signal
max_corr = [np.max(np.abs(corr)) for corr in [corr_s1_sh1, corr_s1_sh2, corr_s1_sh3, corr_s2_sh1, corr_s2_sh2, corr_s2_sh3, corr_s3_sh1, corr_s3_sh2, corr_s3_sh3]]

# Creating a correlation matrix
cross_corr_matrix = np.array(max_corr).reshape(3, 3)

# Finding the maximum cross-correlation for each separated signal
max_corr_indices = np.argmax(cross_corr_matrix, axis=1)

# Converting the max correlation indices to list
max_corr_indices = list(max_corr_indices)

# Creating a list of the separated signals
separated_signals = [sh1, sh2, sh3]

# Reordering separated signals according to the maximum cross-correlation
sh1_reordered = separated_signals[max_corr_indices[0]]
sh2_reordered = separated_signals[max_corr_indices[1]]
sh3_reordered = separated_signals[max_corr_indices[2]]


#fs1 sampling frequency and M is the length of the signals
time = np.arange(M) / fs1

# Create a figure with subplots
fig, axs = plt.subplots(3, 2, figsize=(10,10))

# s1 and sh1
axs[0, 0].plot(time, s1_in)
axs[0, 0].set_title('Original Signal s1')
axs[0, 0].set_xlabel('Time (s)')
axs[0, 0].set_ylabel('Amplitude')
axs[0, 1].plot(time, sh1_reordered)
axs[0, 1].set_title('Separated Signal sh1')
axs[0, 1].set_xlabel('Time (s)')
axs[0, 1].set_ylabel('Amplitude')

# s2 and sh2
axs[1, 0].plot(time, s2_in)
axs[1, 0].set_title('Original Signal s2')
axs[1, 0].set_xlabel('Time (s)')
axs[1, 0].set_ylabel('Amplitude')
axs[1, 1].plot(time, sh2_reordered)
axs[1, 1].set_title('Separated Signal sh2')
axs[1, 1].set_xlabel('Time (s)')
axs[1, 1].set_ylabel('Amplitude')

# s3 and sh3
axs[2, 0].plot(time, s3_in)
axs[2, 0].set_title('Original Signal s3')
axs[2, 0].set_xlabel('Time (s)')
axs[2, 0].set_ylabel('Amplitude')
axs[2, 1].plot(time, sh3_reordered)
axs[2, 1].set_title('Separated Signal sh3')
axs[2, 1].set_xlabel('Time (s)')
axs[2, 1].set_ylabel('Amplitude')

# Add some space between subplots
plt.subplots_adjust(hspace=0.5)

# Show the figure
plt.show()


# Create pandas DataFrame from numpy array
df = pd.DataFrame(corr_matrix)

# Print the DataFrame
print(df)


# MSE between original signal s1 and separated signal sh1
mse_s1_sh1 = mean_squared_error(s1_in, sh1_reordered)
print(f'MSE between s1 and sh1: {mse_s1_sh1}')

# MSE between original signal s2 and separated signal sh2
mse_s2_sh2 = mean_squared_error(s2_in, sh2_reordered)
print(f'MSE between s2 and sh2: {mse_s2_sh2}')

# MSE between original signal s3 and separated signal sh3
mse_s3_sh3 = mean_squared_error(s3_in, sh3_reordered)
print(f'MSE between s3 and sh3: {mse_s3_sh3}')

# write separated signals back to .wav files
sf.write('report2_wav 2/sh1.wav', sh1_reordered, fs1)
sf.write('report2_wav 2/sh2.wav', sh2_reordered, fs1)
sf.write('report2_wav 2/sh3.wav', sh3_reordered, fs1)