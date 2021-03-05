#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

import numpy as np

import scipy.stats as s 

import matplotlib.pyplot as plt 

import seaborn as sns


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


class GaussianNB:
    
    """Instantiate a Gaussian Naive Bayes Object with the following parameters: 
        
        features :               A dataframe consisting of continuous features, excluding labels
        labels :                 A series consisting of binary labels
        train_cv_test_split :    A tuple consisting of fraction for training, cross validation and testing data
        apply_pca :              Boolean value specifying whether to apply PCA or not
        n_principal_components : Number of Principal Components (Eigen vectors having non zero values to keep) 
    """
    
    def __init__(self,features,labels,train_cv_test_split,apply_pca,n_principal_components):
        
        self.unique_labels = list(labels.unique())
        
        self.labels = np.array(labels).reshape(labels.shape[0],1)
        
        self.train_cv_test_split = train_cv_test_split
        
        self.n_principal_components = n_principal_components
        
        if apply_pca == True:
            
            self.X_new = self.apply_dim_reduction(features,self.n_principal_components)
            
            
            
            
    def apply_dim_reduction(self,data,n_components):
        
        X = np.array(data)
        
        X_dash = X - np.mean(X,axis=0).reshape(-1,X.shape[1])
        
        sigma_hat = (1/data.shape[0])*np.matmul(X_dash.T,X_dash)
        
        sigma_hat_decompose = np.linalg.svd(sigma_hat)
        
        Q = sigma_hat_decompose[0]
        
        Q_tilda = Q[:,0:n_components]
        
        X_new = np.matmul(X_dash,Q_tilda)
        
        return X_new
    
    
    
            
    def data_splitting(self):
        
        new_data = pd.DataFrame(data=self.X_new)
        
        new_data['label'] = self.labels
        
        
        
        training_data_len = int(self.train_cv_test_split[0]*new_data.shape[0])
        
        neg_training_data = new_data[new_data['label'] == self.unique_labels[0]].iloc[0:training_data_len//2]
        
        pos_training_data = new_data[new_data['label'] == self.unique_labels[1]].iloc[0:training_data_len//2]
        
        training_data = pd.concat([neg_training_data,pos_training_data])
        
        
        
        
        neg_remain_data = new_data[new_data['label'] == self.unique_labels[0]].iloc[training_data_len//2:]
        
        pos_remain_data = new_data[new_data['label'] == self.unique_labels[1]].iloc[training_data_len//2:]
        
        remaining_data = pd.concat([neg_remain_data,pos_remain_data])
        
        
        
        cv_data_len = int(self.train_cv_test_split[1]*new_data.shape[0])
        
        cv_data = remaining_data.iloc[0:cv_data_len]
        
        testing_data = remaining_data.iloc[cv_data_len:]
        
        return training_data,cv_data,testing_data
    
    
    
    
    def fit(self,data):
        
        mu_hat_neg = np.array(data[data['label'] == self.unique_labels[0]].iloc[:,0:self.n_principal_components].mean())

        sigma_hat_neg = np.array(data[data['label'] == self.unique_labels[0]].iloc[:,0:self.n_principal_components].cov())
        
        
        
        mu_hat_pos = np.array(data[data['label'] == self.unique_labels[1]].iloc[:,0:self.n_principal_components].mean())

        sigma_hat_pos = np.array(data[data['label'] == self.unique_labels[1]].iloc[:,0:self.n_principal_components].cov())
        
        
        
        self.neg_likelihood_params = (mu_hat_neg,sigma_hat_neg)
        
        self.pos_likelihood_params = (mu_hat_pos,sigma_hat_pos)
        
        
        
        
    def evaluate(self,data):
        
        inputs = np.array(data.iloc[:,0:self.n_principal_components])
    
        posterior_neg = s.multivariate_normal.pdf(inputs,self.neg_likelihood_params[0],self.neg_likelihood_params[1])
    
        posterior_pos = s.multivariate_normal.pdf(inputs,self.pos_likelihood_params[0],self.pos_likelihood_params[1])
    
        predicted_category = pd.Series(posterior_pos > posterior_neg)
    
        predicted_category.replace(to_replace=[False,True],value=self.unique_labels,inplace=True)
    
        predicted_results = np.array(predicted_category)
        
        actual_results = np.array(data['label'])
        
        print(classification_report(actual_results,predicted_results,target_names=self.unique_labels))


# In[ ]:


if __name__ == "__main__":
    
    print("Going to run the module as a script")


# In[ ]:




