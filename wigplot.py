# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:41:10 2024

@author: janal
"""

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'xtick.labelsize': '18',
    'ytick.labelsize': '18',
    'legend.fontsize': '18',
    'axes.labelsize': '18'
})

# Read the CSV file
csv_file = './wigner_s1.csv'
df = pd.read_csv(csv_file)

# Extract data from DataFrame
x1vec = df.iloc[:,0].values
p1vec = df.iloc[:,0].values
W_x1p1 = df.iloc[:, 1:].values  # Assuming Wigner data starts from the third column


# Plotting 2D Wigner Function
# plt.figure(figsize=(10, 8))
plt.contourf(x1vec, p1vec, W_x1p1, cmap='viridis')
# plt.colorbar(label='Wigner Function Value')
plt.colorbar()
plt.xlabel(r'X_1')
plt.ylabel(r'P_1')
# plt.title('2D Wigner Function Plot')
plt.tight_layout()
plt.savefig("wigner_s1.png")
plt.show()

# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, W_x1p1, cmap='viridis')

# ax.set_xlabel('x1')
# ax.set_ylabel('p1')
# ax.set_zlabel('Wigner Function Value')
# ax.set_title('3D Wigner Function Plot')

# plt.show()







