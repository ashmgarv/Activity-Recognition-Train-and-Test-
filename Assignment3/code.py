#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import math
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import os


# In[10]:


MyoData_dir = os.listdir("data/MyoData")
groundTruth_dir = os.listdir("data/groundTruth")


# In[11]:


FinalData_np = np.zeros((0,130))


# In[12]:


no_of_users = len(MyoData_dir)


# In[13]:


for u_i, user in enumerate(MyoData_dir):
#     print(user)
    for act in ['fork', 'spoon']:
#         print(act)
        emg_path = ""
        imu_path = ""
        gt_path = ""
        files = os.listdir("data/MyoData/"+ user+"/"+act)
        for file in files:
            if(file[-7:] == "EMG.txt"):
                emg_path = "data/MyoData/"+ user+"/"+act + "/"+file
                break
        for file in files:
            if(file[-7:] == "IMU.txt"):
                imu_path = "data/MyoData/"+ user+"/"+act + "/"+file
                break
        files_gt = os.listdir("data/groundTruth/" + user + "/" + act)
        gt_path = "data/groundTruth/" + user + "/" + act + "/" +files_gt[0]
        
        emg = pd.read_csv(emg_path, header=None, error_bad_lines=False)
        imu = pd.read_csv(imu_path, header=None, error_bad_lines=False)
        gt = pd.read_csv(gt_path, header=None, error_bad_lines=False)
#         emg_imu = pd.merge(emg, imu, on=0, how='outer')
#         import math
        total_idx = gt.shape[0]
        train_idx = int(math.ceil(0.6*total_idx))
#         print("aaaaaaaaaa:  ", total_idx, train_idx)
        for i in range(gt.shape[0]):
#             print("bbbbbb : ", i, train_idx, total_idx)
#             print("bbbbbb: ", user, act, i, train_idx, total_idx)
            try:
                st = int(gt.iloc[i,:2][0])
                end = int(gt.iloc[i,:2][1])
#                 print(st,end, type(st), type(end))
                st = st/30*50
                end = end/30*50
                st = math.floor(st)
                end = math.ceil(end)
                slice_emg = emg.iloc[st:end+1, 1:]
                slice_imu = imu.iloc[st:end+1, 1:]
                user_act_inf = np.array([user,act,i])
                a = np.nanmean(slice_emg, axis=0)
                b = np.nanmean(slice_imu, axis=0)
                c =np.nanvar(slice_emg, axis=0)
                d = np.nanvar(slice_imu, axis=0)
                e = np.nanstd(slice_emg, axis=0)
                f = np.nanstd(slice_imu, axis=0)
                g = np.nanmin(slice_emg, axis=0)
                h = np.nanmin(slice_imu, axis=0)
                ii = np.nanmax(slice_emg, axis=0)
                j = np.nanmax(slice_imu, axis=0)
#                 print(np.fft.fft(slice_emg, n=2, axis=0).shape)
#                 print(np.fft.fft(slice_imu, n=2, axis=0).shape)
                k = np.fft.fft(slice_emg, n=2, axis=0).ravel(order='F').real
                l = np.fft.fft(slice_imu, n=2, axis=0).ravel(order='F').real
#                 print("aaa: ", i, train_idx)
                if(u_i<=17):
                    m = np.array(['train'])
                else:
                    m = np.array(['test'])
#                 all_abc = np.concatenate((user_act_inf, a,b,c,d,e,f,g,h,i,j,k,l))
                all_abc = np.concatenate((user_act_inf, a,b,c,d,e,f,g,h,ii,j,k,l,m))
#                 print(all_abc)
#                 print(slice_emg.shape)
#                 print("\n")
#                 print(slice_imu.shape)
#                 print(FinalData_np.shape)
                
                all_abc = all_abc.reshape((1,all_abc.shape[0]))
#                 print(all_abc.shape)
                FinalData_np = np.concatenate((FinalData_np, all_abc), axis=0)
#                 FinalData_np.append(all_abc)
#                 slice_.to_csv("data/data_sliced/" + user + "/" + str(act) + str(i) +".csv", index=False)
            except:
                st = gt.iloc[i,:2][0]
                end = gt.iloc[i,:2][1]
                print("ERROR: ", user, act, i, st, end, type(st), type(end))
#     print("\n")


# In[14]:


FinalData_np.shape


# In[15]:


cols = ['user', 'act', 'act_i']
cols_feat = []
for i in range(126):
    cols_feat.append("feat_"+str(i))
cols = cols + cols_feat + ['isTrain']
FinalData = pd.DataFrame(FinalData_np, columns = cols)


# In[18]:


# np.mean(FinalData['isTrain'] =='train')


# In[19]:


from sklearn.decomposition import PCA
pca = PCA(n_components=20)
FinalData_pca = pca.fit_transform(FinalData[cols_feat])


# In[20]:


FinalData_pca.shape


# In[21]:


cols_feat = []
for i in range(20):
    cols_feat.append("pca"+str(i))
FinalData_pca_df = pd.DataFrame(FinalData_pca, columns = cols_feat)


# In[22]:


FinalData_with_PCA = pd.concat((FinalData, FinalData_pca_df), axis=1)


# In[ ]:





# In[23]:


FinalData_with_PCA_train = FinalData_with_PCA[FinalData_with_PCA['isTrain']=='train']
FinalData_with_PCA_test = FinalData_with_PCA[FinalData_with_PCA['isTrain']=='test']


# In[24]:


FinalData_with_PCA_test['label'] = (FinalData_with_PCA_test['act'] == 'fork')+0


# In[25]:


cols_feat = []
for i in range(126):
    cols_feat.append("feat_"+str(i))
trainX = FinalData_with_PCA_train[cols_feat]
testX = FinalData_with_PCA_test[cols_feat]


# In[26]:


trainY = (FinalData_with_PCA_train['act']=='fork')+0
testY = (FinalData_with_PCA_test['act']=='fork')+0


# In[ ]:





# ## Decision Tree Classifier

# In[27]:


results = []


# In[28]:


print("Using Decision Tree")
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

parameters = {'criterion':('gini', 'entropy'), 'max_depth':[4, 5, 7, 9, 11]}
clf_dt = DecisionTreeClassifier(random_state=0)
clf_dt_gs = GridSearchCV(clf_dt, parameters, cv=4)
clf_dt_gs.fit(trainX, trainY)

dt_pred = clf_dt_gs.predict(testX)
FinalData_with_PCA_test['dt_predicted'] = dt_pred
print("For Decision Tree")
print("precision_score: ", precision_score(testY, dt_pred))
print("recall_score: ", recall_score(testY, dt_pred))
print("f1_score: ", f1_score(testY, dt_pred))
for gp in FinalData_with_PCA_test.groupby(['user']):
    results.append([gp[0], 'decision_tree', precision_score(gp[1]['label'], gp[1]['dt_predicted']), recall_score(gp[1]['label'], gp[1]['dt_predicted']), f1_score(gp[1]['label'], gp[1]['dt_predicted'])])


# In[29]:


clf_dt_gs.best_params_


# In[30]:


results


# In[ ]:





# In[31]:


results_nn = []


# ## Fully connected Neural Network

# In[32]:


print("Using MLPClassifier")
from sklearn.neural_network import MLPClassifier  
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import GridSearchCV

scaler = StandardScaler()  
scaler.fit(trainX)

trainX_scaled = scaler.transform(trainX)  
testX_scaled = scaler.transform(testX) 

parameters = {'hidden_layer_sizes':[(12,5),(10,5),(8,5)], 'activation':['relu', 'logistic'], 
                 'solver': ['adam'], 'learning_rate_init': [0.001, 0.003, 0.006, 0.01],
              'max_iter': [1000, 1500], 
              'random_state': [1]
             }

mlp = MLPClassifier()  
clf_mlp_gs = GridSearchCV(mlp, parameters, cv=4)
clf_mlp_gs.fit(trainX_scaled, trainY.values.ravel()) 

nn_pred = clf_mlp_gs.predict(testX_scaled)
FinalData_with_PCA_test['nn_predicted'] = nn_pred

print("For NN")
print("precision_score: ", precision_score(testY, nn_pred))
print("recall_score: ", recall_score(testY, nn_pred))
print("f1_score: ", f1_score(testY, nn_pred))

for gp in FinalData_with_PCA_test.groupby(['user']):
    results_nn.append([gp[0], 'neural-net', precision_score(gp[1]['label'], gp[1]['nn_predicted']), recall_score(gp[1]['label'], gp[1]['nn_predicted']), f1_score(gp[1]['label'], gp[1]['nn_predicted'])])
    


# In[33]:


results_nn


# In[34]:


clf_mlp_gs.best_params_


# ## SVM Classifier

# In[35]:


results_svm =[]


# In[36]:


print("Using SVM")
from sklearn.model_selection import GridSearchCV

parameters = {'C':[1.0, 0.8], 'kernel':['linear', 'rbf'], 'gamma':['auto', 'scale']}

from sklearn.svm import SVC
clf_svm = SVC()


clf_svm_gs = GridSearchCV(clf_svm, parameters, cv=4)
clf_svm_gs.fit(trainX_scaled, trainY.values.ravel()) 

svm_pred = clf_svm_gs.predict(testX_scaled)
FinalData_with_PCA_test['svm_predicted'] = svm_pred

print("For SVM")
print("precision_score: ", precision_score(testY, svm_pred))
print("recall_score: ", recall_score(testY, svm_pred))
print("f1_score: ", f1_score(testY, svm_pred))

for gp in FinalData_with_PCA_test.groupby(['user']):
   results_svm.append([gp[0], 'svm', precision_score(gp[1]['label'], gp[1]['svm_predicted']), recall_score(gp[1]['label'], gp[1]['svm_predicted']), f1_score(gp[1]['label'], gp[1]['svm_predicted'])])
   


# In[37]:


clf_svm_gs.best_params_


# In[38]:


results_svm


# In[39]:


results_assignment3 = pd.DataFrame(results+results_nn+results_svm, columns=['user', 'algo', 'precision', 'recall', 'f1'])


# In[40]:


results_assignment3.to_csv("results_assignment3.csv", index=False)


# In[ ]:




