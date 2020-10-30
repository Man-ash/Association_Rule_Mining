#!/usr/bin/env python
# coding: utf-8

# # Association Rule Mining

# In[2]:


from IPython.display import Image
Image(filename='pseudocode.png') 


# In[3]:


Image(filename='image.png') 


# In[4]:


Image(filename='advantages&disadvantages.png') 


# # Demo:
# 
# 
# ## Apriori on a Toy DataSet:

# In[5]:


get_ipython().system(' pip install mlxtend')


# In[6]:


def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]


# ### Apriori Algorithm from scratch

# In[7]:


def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))


# In[13]:


def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not can in ssCnt: ssCnt[can] = 1
                else: ssCnt[can] += 1
                    
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        supportData[key] = support
        if support >= minSupport:
            retList.insert(0,key)
    return retList,supportData


# In[9]:


dataSet = loadDataSet()
dataSet


# In[10]:


C1 = createC1(dataSet)
C1


# In[11]:


D = list(map(set,dataSet))
D


# In[16]:


L1,suppDat0 = scanD(D,C1,0.6)
L1


# In[17]:


suppDat0


# In[18]:


def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])  # set union
    return retList


# In[19]:


def apriori(dataSet, minSupport):
    C1 = createC1(dataSet)
    D = list(map(set,dataSet))
    L1, supportData = scanD(D,C1,minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2],k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


# In[27]:


def generateRules(L, supportData, minConf):
    bigRuleList = []
    for i in range(1,len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if i>1:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList,minConf)
            else:
                prunedH = calcConf(freqSet,H1,supportData,bigRuleList,minConf)
    return bigRuleList                


# In[28]:


def calcConf(freqSet, H, supportData,brl,minConf):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet-conseq, " --> ", conseq, "conf: ", conf)
            brl.append((freqSet-conseq , conseq, conf))
            prunedH.append(conseq)
    return prunedH


# In[29]:


def rulesFromConseq(freqSet, H, supportData, brl, minConf):
    m = len(H[0])
    if (len(freqSet) > (m+1)):
        Hmp1 = aprioriGen(H,m+1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData,brl,minConf)
        
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData,brl,minConf)


# In[36]:


L, suppData = apriori(dataSet,minSupport=0.5)


# In[39]:


rules = generateRules(L, suppData, minConf = 0.6)


# In[40]:


rules


# ## Apriori Algorithm on Bread-Basket Data Set

# In[41]:


import pandas as pd
import numpy as np


# In[48]:


df = pd.read_csv("./BreadBasket_DMS.csv",delimiter=',')
df.head()


# In[49]:


df.info()


# In[50]:


df.columns


# In[51]:


df.loc[df['Item']=='NONE',:]


# In[52]:


df.drop(df.loc[df['Item']=='NONE',:].index,axis=0,inplace=True)


# In[53]:


hot_encoded_df = df.groupby(['Transaction','Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Transaction')


# In[55]:


hot_encoded_df.head()


# In[56]:


def encode_units(x):
    if x <= 0:
        return 0
    else:
        return 1
hot_encoded_df = hot_encoded_df.applymap(encode_units)


# In[57]:


hot_encoded_df.head()


# In[58]:


from mlxtend.frequent_patterns import apriori, association_rules


# In[59]:


freq_itemsets = apriori(hot_encoded_df, min_support=0.01,use_colnames = True)


# In[60]:


rules = association_rules(freq_itemsets, metric='lift')
rules.head(10)


# In[62]:


rules[ (rules['lift'] > 1.1) & (rules['confidence'] > 0.6)]


# In[ ]:




