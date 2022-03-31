#!/usr/bin/env python
# coding: utf-8

# In[244]:


#callum pender 31.3.22

#libraries
# Import nn.functional
import torch.nn.functional as F

import yfinance as yf  
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt

import numpy as np

import torch
import pandas as pd
from sklearn.metrics import mean_squared_error
import torch.nn as nn
import matplotlib

import matplotlib.pyplot as plt

# Get the data for the SPY ETF by specifying the stock ticker, start date, and end date
EndDate = '2022-09-28'
years=365*5
extrap_len=30
Date = datetime.strptime(EndDate, "%Y-%m-%d")
Start = Date - timedelta(days=years)
Forward_facing=Date+timedelta(days=extrap_len)
StartDate=str(Start.year)+'-'+str(Start.month)+'-'+str(Start.day)
ProjectedDate=str(Forward_facing.year)+'-'+str(Forward_facing.month)+'-'+str(Forward_facing.day)

data = yf.download('SPY',StartDate,str(EndDate))
data2 = yf.download('SPY',StartDate,str(ProjectedDate))

# Plot the close prices - adjust to USD
data["Adj Close"].plot()
plt.show()
Forward_facing
data2
data2
ProjectedDate


# In[245]:


datetime.now().year


# In[246]:


split=0.5

acc_list = np.array(data['Adj Close'].values)

train1=acc_list[:int(split*len(acc_list))]
train2=acc_list[int(split*len(acc_list)):len(acc_list)]


# In[247]:


#define periodic function
def xSin(x): #https://arxiv.org/pdf/2006.08195.pdf xSin(x) periodic activation from Tokyo paper
  a=50#factor a controls the periodic frequency, raise for higher f
  return x + torch.sin(a*x)*torch.sin(a*x)/a # periodic activation function 

#define NN
class Net(nn.Module):
    # layers
    def __init__(self):
        super().__init__()
        #layer system could be optimised from here
        self.linear0 = nn.Linear(1, 512)
        self.linear1 = nn.Linear(512, 256)
        self.act1 = xSin # Activation function
        self.linear2 = nn.Linear(256, 128)
        self.linear2b = nn.Linear(128, 64)
        self.linear2c = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 1, bias=True)

    # forward computation
    def forward(self, x):
        x = self.linear0(x)
        x = self.act1(x) 
        x = self.linear1(x)
        x = self.act1(x) 
        x = self.linear2(x)
        x = self.act1(x) 
        x = self.linear2b(x)
        x = self.act1(x) 
        x = self.linear2c(x)
        x = self.act1(x) 
        #x = self.linear2d(x)
        #x = self.act1(x) 
        x = self.linear3(x)
        return x
    


# In[ ]:





# In[248]:


y_full = []

#fraction of total results used for testing
test_fraction=0.01
#number of records used in testing
test_length=int(len(acc_list)*test_fraction)
#should potentially add a drop factor here to reduce overfitting
def fit(acc_list):
    for i in range(0,1):
      ## running for multiple times
      model = Net()


      X = torch.Tensor(range(1, len(acc_list)+1)).float().unsqueeze(1)
      C = torch.max(X)
      X = X / torch.max(X)

      Y = torch.Tensor(acc_list).unsqueeze(1)

      Y = Y / torch.max(Y)
    
      opt = torch.optim.SGD(model.parameters(), 2e-2, momentum=0.9)
      loss_fn = F.mse_loss
      Epoch = 4000
      losses = []
      test_losses = []

      for i in range(Epoch):
        model.train()
        opt.zero_grad()
        y = model(X[1:len(acc_list)-test_length])#training set

        loss = loss_fn(y, Y[:len(acc_list)-test_length-1])

        (loss).backward()
        opt.step()
        losses.append(loss)


        if i % 100 == 0:
          print(loss)
          x_test = torch.Tensor(range(1, len(acc_list)+extrap_len) ).float().unsqueeze(1) /C

          #plt.axvline(x=len(acc_list)-test_length)
          #plt.axvline(x=len(acc_list))


          y_test = model(x_test)
      y_full.append(y_test.detach())
    return(y_test)

#normalise to original signal using custom boundaries
def normalise(acc_list,Y):
    Y=Y.detach().numpy()
    ref_Y=min(acc_list)+(Y-min(Y))/(max(Y)-min(Y))*(max(acc_list)-min(acc_list))
    return(ref_Y)

Y=fit(acc_list)
Y=normalise(acc_list,Y)


# In[249]:


#plot 5 years
x_new=np.arange(len(Y)-extrap_len,len(Y),1)
plt.figure(figsize=(15,10))
plt.grid()
plt.title('30-Day Stock Index Series Extrapolation\n5 Year Plot\nDecember 2021 - January 2022')
plt.plot(Y[0:len(Y)-extrap_len],label='Model Fitting',linestyle='--')
plt.plot(x_new,Y[len(Y)-extrap_len:len(Y)],label='One Month Forward Prediction')
plt.xlabel('Date')
plt.ylabel('S&P 500 Index/10')
plt.plot(np.array(data2['Adj Close'].values), label='S&P 500')

plt.axvline(x=len(Y)-extrap_len)
#plt.axvline(x=len(Y))
plt.xticks([0, len(Y)-extrap_len, len(Y)],[str(StartDate), str(EndDate), str(ProjectedDate)],rotation=70)  
plt.legend()
plt.tight_layout()
fig = plt.gcf()
#fig.savefig('Aug_5y'+'.png', bbox_inches='tight',facecolor='w')


# In[250]:


#plot 1 year
plt.figure(figsize=(15,10))
plt.plot(Y[len(Y)-365:len(Y)-extrap_len],label='Model Fitting',linestyle='--')
x_new_new=np.arange(365-extrap_len,365,1)
plt.plot(x_new_new,Y[len(Y)-extrap_len:len(Y)],label='One Month Forward Prediction')
plt.grid()
plt.xlabel('Date')
plt.ylabel('S&P 500 Index/10')
plt.title('30-Day Stock Index Series Extrapolation\n1 Year Plot\nDecember 2021 - January 2022')
plt.plot(np.array(data2['Adj Close'].values[len(data2['Adj Close'].values)-365:]),label='S&P 500')
plt.axvline(x=365-extrap_len)
print(extrap_len)
plt.xticks([0, 365-extrap_len, 365],['2020-12-28', str(EndDate)+'\nExtrapolation\nOnwards', str(ProjectedDate)],rotation=70)  
plt.legend()
#plt.savefig('Dec1y'+'.png', bbox_inches='tight')
fig = plt.gcf()

#fig.savefig('Aug_1y'+'.png', bbox_inches='tight',facecolor='w')


# In[ ]:





# In[251]:


model_snp=Y[len(Y)-extrap_len:len(Y)]
real_snp=data2['Adj Close'][len(data2['Adj Close'])-extrap_len:len(data2['Adj Close'])]

minmodel=min(model_snp)
maxmodel=max(model_snp)
avemodel=np.mean(model_snp)
minreal=min(real_snp)
maxreal=max(real_snp)
avereal=np.mean(real_snp)

range_frame=str(float(avemodel))+','+str(float(maxmodel))+','+str(float(minmodel))+','+str(float(avereal))+','+str(float(maxreal))+','+str(float(minreal))

plt.plot(avemodel,'o',color='red')
plt.plot(maxmodel,'x',color='red')
plt.plot(minmodel,'x',color='red')
plt.plot(minreal,'x',color='blue')
plt.plot(maxreal,'x',color='blue')
plt.plot(avereal,'o',color='blue')


#save results to file 
#f= open(str(EndDate)+"_Range.txt","w+")
#f.write(range_frame)
#f.close()


# In[252]:


#MSE
rms = mean_squared_error(Y[len(Y)-extrap_len:len(Y)],data2['Adj Close'][len(data2['Adj Close'])-extrap_len:len(data2['Adj Close'])].values, squared=False)

print(rms)

def mean_absolute_percentage_error_(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


percentage=mean_absolute_percentage_error_(real_snp, model_snp)

#save results to file 
#f= open(str(EndDate)+"_Error.txt","w+")
#errors=str(float(rms))+','+str(float(percentage))
#f.write(str(errors) + '\n')
#f.close()


# In[ ]:




