#!/usr/bin/env python
# coding: utf-8

# In[10]:


import nltk
import pandas as pd
import numpy as np
import re
from langdetect import detect
from nltk.corpus import stopwords
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords


# In[11]:


df = pd.read_csv('dataSet\jobs_data.csv')


# In[12]:


df.head(10)


# In[13]:


df.drop('Unnamed: 0' ,axis = 1 ,inplace = True)


# In[14]:


df.tail()


# In[15]:


df.shape


# In[16]:


df.info()


# In[17]:


df['title'].value_counts( )


# In[18]:


len([df['jobFunction'].value_counts()][0])


# In[19]:


df['jobFunction'].value_counts( )


# In[20]:


df['industry'].value_counts( )


# ### clean Data

# In[21]:


stop = stopwords.words('english')
stop_words_ = set(stopwords.words('english'))
wn = WordNetLemmatizer()

def black_txt(token):
    return  token not in stop_words_ and token not in list(string.punctuation)     
  
def clean_txt(text):
    clean_text = []
    clean_text2 = []
    text = re.sub("'", "",text)
    text=re.sub("(\\d|\\W)+"," ",text) 
    text = text.replace("nbsp", "")
    clean_text = [ wn.lemmatize(word, pos="n") for word in word_tokenize(text.lower()) if black_txt(word)]
    clean_text2 = [word for word in clean_text if black_txt(word)]
    return " ".join(clean_text2)


# In[22]:


df['jobFunctionTxt'] = df['jobFunction'].apply(clean_txt)
df['titleTxt'] = df['title'].apply(clean_txt)
df['industryTXT'] = df['industry'].apply(clean_txt)


# In[23]:


df.head()


# In[24]:


df['titleTxt'].value_counts( )


# In[25]:


df.iloc[93]


# In[26]:


df['lang'] = df['title'].apply(detect) #detect lang of title column


# In[27]:


df


# In[28]:


df['lang'].value_counts( )


# In[29]:


df_AR = df.loc[df['lang'] == 'ar']
df_AR.head()


# ### Translate Arabic title to English

# In[30]:


from textblob import TextBlob
def translate_AR_EN(word0):
    word = TextBlob('{0}'.format(word0) )
    try:
        e = word.translate(from_lang = 'ar' , to = 'en')
    except:
        e = word
    return e


# In[31]:


df_AR['titleTxt'] = df_AR['titleTxt'].apply(translate_AR_EN)


# In[32]:


df_AR.head()


# In[33]:


df_AR['titleTxt' ] = df_AR['titleTxt' ].apply(''.join)


# In[34]:


df_AR.head()


# In[35]:


df.loc[df['lang'] == 'ar'].shape


# In[36]:


df_AR.shape


# In[37]:


df.loc[df['lang'] == 'ar'] = df_AR


# In[38]:


df


# In[39]:


df.loc[df['lang'] == 'ar']


# In[40]:


df.drop('lang' ,axis = 1 ,inplace = True)


# In[41]:


df.head()


# ### Handling Rows With NAN Value
# *we have nan value at jopFunction column to handle it:*
# if this jobFunction 'which has nan' its title value only one in df drop it,
# else search about title(1) has the same title'which its jobFunction has nan' and
# assign jobFunction'which has nan' by jobFunction of title(1)
# 
# 

# In[42]:


df['jobFunction'].loc[df['jobFunctionTxt'] == 'nan'].count() #count of nan value in dataset


# In[43]:


df.loc[df['jobFunctionTxt'] == 'nan']


# In[44]:


df['jobFunctionTxt'].replace('nan', np.nan , inplace = True)
null_df = df.loc[df['jobFunctionTxt'].isnull()]
null_df.head()


# In[45]:


df.info()


# In[46]:


df.dropna(inplace = True)
df.info()


# In[47]:


null_df.shape


# In[48]:


df['titleTxt'].iloc[1467-116]


# In[49]:


l=null_df['industry'].isin(df['industry'])


# In[50]:


null_df['eq'] = l


# In[51]:


null_df.head()


# In[52]:


null_df['eq'].value_counts( )


# In[53]:


null_df = null_df[null_df['eq'] != False]


# In[54]:


null_df


# In[55]:


nullV = null_df['industryTXT'].values
dfV = df['industryTXT'].values
len(dfV)


# In[56]:


x= np.empty((10653,), dtype =str)
nullV1 = np.append(nullV,x)


# In[57]:


dfV.shape == nullV1.shape


# In[58]:


arrFlag = np.equal(dfV,nullV1)
arrFlag


# In[59]:


df['flag'] = arrFlag


# In[60]:


df


# In[61]:


df['flag'].value_counts()


# In[62]:


s = [] 
'''
s= 
[index_JopFunction_Has_NaN'e.g:which title value = "x" ',
index_JopFunction_Dont_Has_NaN'e.g:which title value = "x" ']
'''
for inxi ,i in enumerate(dfV):
    for inxj,j in enumerate(nullV):
        if i == j:
            s.append([inxj,inxi])        


# In[63]:


len(s) 


# In[64]:


s = sorted(s)
s


# In[65]:


flag = []
for i in s:
    if i[0] not in flag:
        flag.append(i[0])
        continue
    s.remove(i)


# In[66]:


for i in s:
    null_df['jobFunctionTxt'].iloc[i[0]] = df['jobFunctionTxt'].iloc[i[1]]


# In[67]:


null_df['jobFunctionTxt'].iloc[0]


# In[68]:


null_df['jobFunctionTxt'].iloc[0] == df['jobFunctionTxt'].iloc[34]


# In[69]:


null_df


# In[70]:


df.iloc[34]


# In[71]:


null_df.iloc[98]


# In[72]:


df.iloc[2966]


# In[73]:


null_df.drop('eq' ,axis = 1 ,inplace = True)
#null_df.drop('eq' ,axis = 1 ,inplace = True)


# In[74]:


mergedStuff = pd.merge(df, null_df, how='outer')


# In[75]:


mergedStuff.info()


# In[76]:


df = mergedStuff
df.head()


# In[77]:


df.drop('flag' ,axis = 1 ,inplace = True)


# In[78]:


df.info()


# In[ ]:





# ### Feature Extraction by TF-IDF

# In[79]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
#tf = TfidfVectorizer(analyzmer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
allData = pd.DataFrame()
allData['dfTXT'] =  df['industryTXT']+df['jobFunctionTxt']+df['titleTxt']
tfidf_matrix = tfidf_vectorizer.fit_transform((allData['dfTXT']))
#tfidf_vectorizer.fit_transform((Merge_data['text']))


# ### Calculating Cosine Similarity

# In[80]:


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

cosine_similarities = linear_kernel(tfidf_matrix)
results = {}
for idx, row in df.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], df['titleTxt'][i]) for i in similar_indices]
    results[row['titleTxt']] = similar_items[1:]
    
print('done!')


# ### Recommender System

# In[81]:


def item(word):
    try:
        return df.loc[df['titleTxt'] == word]['jobFunctionTxt'].tolist()[0]
    except:
        print('sorry My Model dont Know U...OUT! '.upper())
# Just reads the results out of the dictionary.
def recommend(word, num):
    try:
        recs = results[word][:num]
        re = []
        g= 1
        for rec in recs:
            re.append("Recommended " +str(g)+':'+ item(rec[1]))
            re.append (" ( score = " + str(rec[0]) + " )")
            g=g+1
        return re
    except:
        print('sorry My Model dont Know U...OUT! '.upper())


# ### Test

# In[82]:


r = recommend("full stack php developer",10)


# In[83]:


r


# # API

# In[ ]:


from sys import platform
import pandas as pd
from flask import Flask,jsonify ,request
app = Flask(__name__)
@app.route('/job/<string:title>/<int:num>', methods=['GET'])
def test(title,num):
    lst = recommend(title,num)
    d = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}  
    return jsonify(d)

app.run(debug=False)


# In[ ]:




