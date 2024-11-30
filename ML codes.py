#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler , StandardScaler , MaxAbsScaler , Normalizer 
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
import sklearn.decomposition as dec
from sklearn.linear_model import SGDRegressor , Ridge , LinearRegression , Lasso , LassoLars ,RANSACRegressor, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor,XGBClassifier
from sklearn.ensemble import AdaBoostRegressor , RandomForestRegressor , GradientBoostingRegressor , ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from math import sqrt
import statistics
import shap
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from matplotlib.cm import get_cmap
from sklearn.metrics import mean_squared_error    
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.font_manager as font_manager
import random
from lazypredict.Supervised import LazyRegressor
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay


# In[ ]:


df = pd.read_excel(r"D:\Articles\Flexural pull_out\Machine - With Alireza - 2024.xlsx",sheet_name='tmax' ,header = 0 )
y = df.iloc[:, 8].to_numpy().reshape((-1, 1))
X = df.iloc[:, [0,1,2,3,4,5,6,7]].to_numpy()
Xtr , Xte , ytr , yte = train_test_split(X, y, train_size=0.7 ,random_state=42)


# In[19]:


# First DecisionTreeRegressor model
df = pd.read_excel(r"D:\Articles\Flexural pull_out\Machine - With Alireza - 2024.xlsx",sheet_name='tmax' ,header = 0 )
y = df.iloc[:, 8].to_numpy().reshape((-1, 1))
X = df.iloc[:, [0,1,2,3,4,5,6,7]].to_numpy()
Xtr , Xte , ytr , yte = train_test_split(X, y, train_size=0.7 ,random_state=42)
model=DecisionTreeRegressor(random_state=0)
model.fit(Xtr, ytr)
yprtr = model.predict(Xtr)
yprte = model.predict(Xte)
r2tr = round(r2_score(ytr , yprtr), 2)
r2te = round(r2_score(yte , yprte), 2)
msetr = round(mean_squared_error(ytr , yprtr)**0.5, 2)
msete = round(mean_squared_error(yte , yprte)**0.5, 2)
maetr = round(mean_absolute_error(ytr , yprtr), 2)
maete = round(mean_absolute_error(yte , yprte), 2)
a = 0
b = 35

# Second DecisionTreeRegressor model 15, 1, 5, 'auto'
Xtr1 , Xte1 , ytr1 , yte1 = train_test_split(X, y, train_size=0.7, random_state=42)
model1=DecisionTreeRegressor(random_state=0, max_depth=9, min_samples_leaf=2, min_samples_split=5, max_features='log2')
model1.fit(Xtr1 , ytr1)
yprtr1 = model1.predict(Xtr1)
yprte1 = model1.predict(Xte1)
r2tr1 = round(r2_score(ytr1 , yprtr1), 2)
r2te1 = round(r2_score(yte1 , yprte1), 2)
msetr1 = round(mean_squared_error(ytr1 , yprtr1)**0.5, 2)
msete1 = round(mean_squared_error(yte1 , yprte1)**0.5, 2)
maetr1 = round(mean_absolute_error(ytr1 , yprtr1), 2)
maete1 = round(mean_absolute_error(yte1 , yprte1), 2)
a1 = 0
b1 = 35

# Plotting the figures
plt.figure(figsize=(12, 6))
font = {'family': 'Times New Roman', 'size': 14}
plt.rc('font', **font)
plt.subplot(1, 2, 1)
plt.scatter(ytr, yprtr, s=80,marker='X', facecolors='mediumturquoise', edgecolors='black',
            label=f'\n Train \n R2 = {r2tr}  \nRMSE = {msetr}\nMAE = {maetr}')
plt.scatter(yte , yprte, s=80, marker='P',facecolors='fuchsia', edgecolors='black',
            label=f'\n Test \n R2 = {r2te} \nRMSE = {msete}\nMAE = {maete}')
plt.plot([a, b], [a, b], c='black', lw=1.4, label='y = x')
plt.title(f'Before applying grid search', fontsize=14)
plt.xlabel('τmax (MPa)_Experimental', fontsize=15)
plt.ylabel('τmax (MPa)_Predicted', fontsize=15)
font = {'family': 'Times New Roman', 'size': 14}
plt.rc('font', **font)
plt.legend(loc=4)
plt.tight_layout()

plt.subplot(1, 2, 2)
plt.scatter(ytr1, yprtr1, s=80,marker='X', facecolors='darkcyan', edgecolors='black',
            label=f'\n Train \n R2 = {r2tr1}  \nRMSE = {msetr1}\nMAE = {maetr1}')
plt.scatter(yte1 , yprte1, s=80, marker='P',facecolors='deeppink', edgecolors='black',
            label=f'\n Test \n R2 = {r2te1} \nRMSE = {msete1}\nMAE = {maete1}')
plt.plot([a1, b1], [a1, b1], c='black', lw=1.4, label='y = x')
plt.title(f'After applying grid search', fontsize=14)
plt.xlabel('τmax (MPa)_Experimental', fontsize=15)
plt.ylabel('τmax (MPa)_Predicted', fontsize=15)
font = {'family': 'Times New Roman', 'size': 14}
plt.rc('font', **font)
plt.legend(loc=4)
plt.tight_layout()

plt.savefig(r"D:\Articles\Flexural pull_out\New Figes\DT.SVG", dpi=1000,format='SVG')
plt.show()


# In[27]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Data for R2 scores
r2_result = [['0.85', '0.86', '0.89', '0.90'],
             ['0.76', '0.83', '0.86', '0.85']]
r2_result = np.array(r2_result, dtype=np.float)

# Data for RMSE
rmse_result = [['2.44', '2.4', '2.1', '2.02'],
               ['3.1', '2.62', '2.38', '2.48']]
rmse_result = np.array(rmse_result, dtype=np.float)

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=250, subplot_kw={'projection': '3d'})

# Plot for R2 scores
ax1 = axs[0]
ax1.set_xlabel('ML models', labelpad=10)
ax1.set_ylabel('Metrics', labelpad=10)
ax1.set_zlabel('R2')

xlabels = np.array(['DT', 'RF', 'GB', 'XGB'])
xpos = np.arange(xlabels.shape[0])
ylabels = np.array(['After', 'Before'])
ypos = np.arange(ylabels.shape[0])

xposM, yposM = np.meshgrid(xpos, ypos, copy=False)
zpos = r2_result
zpos = zpos.ravel()

dx = 0.6
dy = 0.4
dz = zpos

ax1.w_xaxis.set_ticks(xpos + dx / 2.)
ax1.w_xaxis.set_ticklabels(xlabels)

ax1.w_yaxis.set_ticks(ypos + dy / 2.)
ax1.w_yaxis.set_ticklabels(ylabels)

values = np.linspace(0.2, 1., xposM.ravel().shape[0])
colors = cm.viridis(values)

# Add text annotations at the top of each column
for xpos, ypos, zpos, value in zip(xposM.ravel(), yposM.ravel(), dz, zpos):
    ax1.text(xpos, ypos, value, '%.2f' % value, ha='left', va='baseline')

ax1.bar3d(xposM.ravel(), yposM.ravel(), np.zeros_like(dz), dx, dy, dz, color=colors)

# Plot for RMSE
ax2 = axs[1]
ax2.set_xlabel('ML models', labelpad=10)
ax2.set_ylabel('Metrics', labelpad=10)
ax2.set_zlabel('RMSE')

zpos = rmse_result
zpos = zpos.ravel()

values = np.linspace(0.2, 1., xposM.ravel().shape[0])
colors = cm.plasma(values)

# Add text annotations at the top of each column
for xpos, ypos, zpos, value in zip(xposM.ravel(), yposM.ravel(), dz, zpos):
    ax2.text(xpos, ypos, value, '%.2f' % value, ha='center', va='bottom')

ax2.bar3d(xposM.ravel(), yposM.ravel(), np.zeros_like(dz), dx, dy, dz, color=colors)

plt.show()


# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
from math import pi

font = {'family': 'Times New Roman', 'size': 14}
plt.rc('font', **font)

# Set data
df = pd.DataFrame({
    'group': ['BR'],
    'DT': [0.99],
    'XGB': [0.99],
    'GEP': [0.94],
})


# number of variables
categories = list(df)[1:]
N = len(categories)

# Values for the radar chart
values = df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]

# Angles for each axis
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Initialize the radar plot
ax = plt.subplot(111, polar=True)

# Draw one axis per variable and add labels
plt.xticks(angles[:-1], categories, color='black', size=10)

# Draw y-labels
ax.set_rlabel_position(0)
plt.yticks([0.8,0.85,0.90,0.95], ["0.8", "0.85","0.9","0.95"], color="black", size=9,ha='right')
plt.ylim(0.8, 1)

# Plot data with specified color and linewidth
ax.plot(angles, values, linewidth=2, linestyle='solid', color='blue')

# Fill the area with specified color and transparency
ax.fill(angles, values, 'blue', alpha=0.1)


# Show the radar chart
plt.savefig(r"D:\Articles\Flexural pull_out\New Figes\R2.SVG", dpi=1000,format='SVG')
plt.show()


# In[3]:


df = pd.read_excel(r"D:\Articles\Flexural pull_out\Machine - With Alireza - 2024.xlsx",sheet_name='tmax' ,header = 0 )
y = df.iloc[:, 9].to_numpy().reshape((-1, 1))
X = df.iloc[:, [0,1,2,3,4,5,6,7]].to_numpy()



Xtr , Xte , ytr , yte = train_test_split(X, y, train_size=0.6 ,random_state=16)

# First DecisionTreeRegressor model
model=XGBClassifier()
model.fit(Xtr, ytr)
yprtr= model.predict(Xtr)
yprte= model.predict(Xte)
cm1=confusion_matrix(ytr, yprtr)
cm2=confusion_matrix(yte, yprte)
fig, axes=plt.subplots(1,2, figsize=(12,6))

disp1= ConfusionMatrixDisplay(confusion_matrix=cm1)
disp1.plot(ax=axes[0])
axes[0].set_title('Train')

disp2= ConfusionMatrixDisplay(confusion_matrix=cm2)
disp2.plot(ax=axes[1])
axes[1].set_title('Test')
plt.savefig(r"D:\Articles\Flexural pull_out\New Figes\CM.SVG", dpi=1000,format='SVG')
plt.show()

acc= accuracy_score(yte,yprte )
acc


# In[4]:


from sklearn.metrics import accuracy_score, classification_report
print(classification_report(yte,yprte ))


# In[ ]:





# In[ ]:




