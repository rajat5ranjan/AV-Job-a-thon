import pandas as pd
import numpy as np
from scipy.stats import rankdata
from scipy.special import softmax


kv1=pd.read_csv('AV-job-cb-sub6.csv') #0.813877542491945
kv2=pd.read_csv('AV-job-stack-sub8.csv') #0.813522186906736

kv1['Response']=(kv1['Response']+kv2['Response'])/2
kv1.to_csv('Sub_v0.1.csv',index=False) #0.813885160363821
#
#kv1.loc[kv1['Response']>=0.8,'Response']=1
#kv1.loc[kv1['Response']<=0.05,'Response']=0
##    sub1.loc[sub1[c]<0.05,c]=0
#
#print(kv1.describe())
#kv1.to_csv('Sub_v0.2.csv',index=False) #0.813885160363821



kv3=pd.read_csv('AV-frk1-job-cb-sub3.csv') #0.815527283335571
kv4=pd.read_csv('Sub_v0.1.csv') #0.813885160363821
#predict_list=[]
#predict_list.append(kv1['Response'].values)
#predict_list.append(kv2['Response'].values)
#predict_list.append(kv3['Response'].values)
#predict_list.append(kv4['Response'].values)
#predictions = np.zeros_like(predict_list[0])
#for predict in predict_list:
#    predictions = np.add(predictions, rankdata(predict)/predictions.shape[0])
#predictions /= len(predict_list)
#
#kv4['Response'] = predictions
#kv4.to_csv('Sub_v0.3_ranked.csv', index=False) #0.814786898740417


kv4['Response']=kv1['Response']*0.1+kv2['Response']*0.1+kv3['Response']*0.7+kv4['Response']*0.1
kv4.to_csv('Sub_v0.4.csv', index=False) #0.815592216624416


kv5=pd.read_csv('AV-job-cb-sub9.csv') #0.0.815731514853
kv6=pd.read_csv('Sub_v0.4.csv') #0.815592216624416
kv7=pd.read_csv('AV-job-stack-sub9.csv') #0.815527936296018


kv7['Response']=kv5['Response']*0.6+kv6['Response']*0.2+kv7['Response']*0.1+kv3['Response']*0.1
kv7.to_csv('Sub_v0.5.csv', index=False) 









