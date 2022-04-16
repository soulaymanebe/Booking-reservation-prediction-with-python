#!/usr/bin/env python
# coding: utf-8

# # DATA MINING

# ### BELHAJ SOULAYMANE - ESSALIHI MOUAD

# #### DATA ine2

# In[101]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import missingno as msno


# # I. Dataset

# In[102]:


data = pd.read_csv(r"hotels.csv", delimiter=',')
data 


# In[103]:


data.head()


# In[104]:


data.dtypes


# In[105]:


# Print info of DataFrame
data.info()


# # II. Data preprocessing

# In[106]:


# Print number of missing values
data.isna().sum()


# In[107]:


msno.matrix(data)
plt.show()


# In[108]:


msno.bar(data)


# In[109]:


#Drop rows having missing values except for variables like Agent or Company

data = data.dropna(subset=['country', 'children']).reset_index(drop=True)


# In[110]:


data.isna().sum()


# In[111]:


# Replace mouths from names to numbers
mapp = {'January': '01', 
            'February': '02',
            'March': '03',
            'April': '04',
            'May': '05',
            'June':'06',
            'July':'07',
            'August':'08',
            'September':'09',
            'October':'10',
            'November':'11',
            'December': '12'}

data['arrival_date_month'] = data['arrival_date_month'].replace(mapp)
data['arrival_date'] = pd.to_datetime(data[['arrival_date_year','arrival_date_month','arrival_date_day_of_month']].apply(
    lambda x: '-'.join(x.dropna().astype(str)),axis=1))
data['arrival_date']


# In[112]:


data.drop(['arrival_date_year','arrival_date_month', 'arrival_date_day_of_month'], axis=1, inplace=True)


# In[113]:


data.head()


# In[114]:


data['reservation_status_date'] = pd.to_datetime(data['reservation_status_date'], format = '%Y-%m-%d')

data['reservation_status_date']


# ### Pour  reservation status check-out reservation_status_date doit etre superieure ou egale a arrival_date

# In[115]:



df = data[data['reservation_status_date']<data['arrival_date']]
df.groupby(['reservation_status'])['arrival_date'].count()


# ### Nous avons aucun Check-Out reservation qui a ete faite avant la date d'arrive

# ### Pour  la reservation qui a ete annuler par le client  reservation_status_date doit etre inferieure  ou egale a arrival_date,  verifions si nous avons une contradiction tel que le date de l'annulation est superieur au date d'arrive

# In[116]:


df1 = data[data['reservation_status'] == 'Canceled'] 
print(df1[df1['reservation_status_date']>df1['arrival_date']]['arrival_date'].count())


# In[117]:


data['customer_type'].unique()


# ### Data Standardization

# In[118]:


from sklearn.preprocessing import StandardScaler
col_names = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights','adults','children','babies',
             'previous_cancellations','previous_bookings_not_canceled','booking_changes','days_in_waiting_list']
col_names_new = ['lead time', 'stays in weekend nights', 'stays in week nights','adults ','children ','babies ',
             'previous cancellations','previous bookings not canceled','booking changes','days in waiting list']
sc = StandardScaler()
scaled_values = sc.fit_transform(data[col_names].values)
scaled_df = pd.DataFrame(scaled_values, columns = col_names_new)
df3 = pd.concat([data, scaled_df], axis=1)
df3.drop(col_names , axis = 1, inplace = True)
dataset = df3.copy()
dataset


# In[119]:



df3 = pd.concat([data, scaled_df], axis=1)
df3.drop(col_names , axis = 1, inplace = True)
df3


# In[120]:


dataset = df3.copy()
dataset


# ## Explanatory Data Analysis

# ##  
# - Dans ce qui suit nous allons concidere la variabe is_cancelled notre variable target 

# 
# - Desctiptive statistics for **datetime variables** 

# In[121]:


dataset.describe(datetime_is_numeric=True)['arrival_date']


# ##   
# - Desctiptive statistics for **Categorical variables** 

# In[122]:


dataset.describe(include=['object'])


# voyons la relations  des categories de chaques variable avec notre variable dependante `is_canceled`
# cette etape va nous donnee une vue generale sur les variables categorique qui seront utile pour notre model, par la suite nous allons effectuer des test pour selectionner parmis ces variables celles qui sont plus importante

# In[123]:


def cancel(columnname) : 
    cancellation=dataset[dataset['is_canceled'] == 1].groupby(dataset[columnname])['is_canceled'].count()
    df=cancellation.to_frame()
    df.reset_index(level=0, inplace=True)
    plt.rcParams["figure.figsize"] = (10,10)
    return (plt.bar(df[columnname], df['is_canceled'], color ='maroon',width = 0.5))
def notcancel(columnname) : 
    cancellation=dataset[dataset['is_canceled'] == 0].groupby(dataset[columnname])['is_canceled'].count()
    df1=cancellation.to_frame()
    df1.reset_index(level=0, inplace=True)
    plt.rcParams["figure.figsize"] = (10,10)
    return (plt.bar(df[columnname], df['is_canceled'], color ='blue',width = 0.5))


# In[124]:


cancel('customer_type')


# In[125]:


cancel('assigned_room_type')


# In[126]:


cancel('market_segment')


# In[127]:


cancel('deposit_type')


# In[128]:


cancel('distribution_channel')


# In[129]:


cancel('reservation_status')


# - comme on peut remarquer, il existe des categorie qui ont plus tendance a annuler la reservations, mais nous ne pouvons pas seulement conclure cela de ces graphes, puisque la frequence des  categories dans une seulle colonne varie.  
# - c'est pour cela que nous devons faire des test pratique qui vont nous reveler le degre d'influence  de chaque  variables  sur la variable `is_canceled` 

# ###  Chi-square test 

# In[130]:


from scipy.stats import chi2_contingency


# In[131]:



dataset['Cancelation'] = dataset['is_canceled'].replace({ 1 : 'Canceled', 
                                      0 : 'NotCanceled'})


# La meilleure facon pour avoir un aperçu pour nos variables categorique est leur vizualisation en utilisant **Heatmaps**

# In[132]:


contigency= pd.crosstab(dataset['Cancelation'], dataset['deposit_type'])
plt.figure(figsize=(12,8))
sns.heatmap(contigency, annot=True, cmap="YlGnBu") 


# In[133]:


features = ['hotel','meal','country','market_segment','distribution_channel','reserved_room_type','assigned_room_type',
            'deposit_type','customer_type','reservation_status']
p_list = []
for i in range(0,10):
    contigency= pd.crosstab(dataset['Cancelation'], dataset[features[i]])
    c, p, dof, expected = chi2_contingency(contigency)
    p_list.append(p)
print(p_list)
    


# #### 
# ### Puisque toute les p-value sont presque nul, cela montre que toutes les features sont statiquement signifiant pour note variable independante. Cependant, il y a surement des feature qui sont plus important que d'autre.
# 
# - Dans ce qui suit nous allons selectionner les meilleures variables (sois 7 meilleure variables par exemple) parmis ces 10 variables ![image.png](attachment:image.png)
# 
# #### 

# In[134]:


encodedDf = dataset[['hotel','meal','country','market_segment','distribution_channel','reserved_room_type',
  'assigned_room_type','deposit_type','customer_type','reservation_status','Cancelation']]
encodedDf


# In[135]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for column_name in encodedDf.columns:
    if encodedDf[column_name].dtype == object:
        encodedDf[column_name] = le.fit_transform(encodedDf[column_name])
    else:
        pass


# In[136]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X = encodedDf[['hotel','meal','country','market_segment','distribution_channel','reserved_room_type','assigned_room_type',
               'deposit_type','customer_type','reservation_status']]
X = X.astype(str)
y = encodedDf[['Cancelation']].astype(str)
selector =  SelectKBest(chi2, k=7)
X_new = selector.fit_transform(X, y)
cols = selector.get_support(indices=True)
features_df_cat= encodedDf.iloc[:,cols]
features_df_cat


# ## Les colonnes si-dessus sont les colonne qui sont les plus statiquement signifiants pour la variable Cancelation 

# - Numerical variables 

# In[137]:


dataset.info()


# In[138]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numdf = dataset.select_dtypes(include=numerics)
numdf


# In[139]:


numdf.corr()


# In[140]:


numdff = numdf.copy()
numdff.drop(['company', 'agent'], axis =1, inplace= True)
from scipy import stats
corrlist = []
plist= []
for columns in numdff.columns:
    corr , p = stats.pointbiserialr(numdff['is_canceled'], numdff[columns])
    corrlist.append(corr)
    plist.append(p)
dict = {'Correlation ' : corrlist, 'P-Value' : plist} 
Summary = pd.DataFrame(data = dict , index = numdff.columns)
Summary


# Dans ce tableau nous avons la correlation des variables numerique avec la variable dependante ainsi que la P-Value
# - Note que nous ne pouvons pas faire point biserialr correlation sur des variable contenant des valeurs Nan, pour les variables 'company' et 'agent' nous allons prendre en consideration pearson correlation, puisque dans notre cas, nous avons pas une difference significatif entre les deux types de correlation
# 
# 

# ## Conclusion 
# apres cette analyse des donnee numerique, et se basant sur P-value et la correlation: nous pouvons deduire que: 
# - nous avons une faible correlation entre tout les variables independante et la variable dependante, cela montre que nous avons une relations de non-linearite 
# - Nous avons des relations qui ont  une signification statistique tres forte(P-value <0.05)
# On peut conclure que les variables suivant n'aurons pas d'impact sur la variable dependante (P-value >0.01): 
# - stays in weekend nights
# - children 

# ## Top 5 most important features that has the strongest relationship with the target variable 
#  - Feature selection is performed using ANOVA F measure via the f_classif() function.

# In[141]:



from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
fs = SelectKBest(score_func=f_classif, k=7)
colnum = numdff.columns.tolist()[1:]
y = numdff['is_canceled']
X= numdff[colnum]
X_selected = fs.fit_transform(X, y)
cols = fs.get_support(indices=True)
features_df_num= numdff.iloc[:,cols]
features_df_num


# In[142]:


cancel('hotel')


# In[143]:



gp = data[data['is_canceled'] == 1]
gp.head()


# In[144]:


ypoints = gp['adults'].unique()


# In[145]:


sns.countplot(data=data, x='adults', hue='is_canceled')


# ### Spliting data to Train/test data 
#  - at first we only gonna train&test the model with the best 14 features ( numerical & categorical) and then compare the performance with all features being used
# 

# In[146]:


from sklearn.model_selection import train_test_split, cross_val_score


# In[147]:


best_features = pd.concat ([features_df_cat, features_df_num], axis = 1)
best_features['churn'] = dataset['is_canceled']
X = np.asarray(best_features.values[:,:14]) 
y = np.asarray(best_features[['churn']])


# In[148]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[149]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

params = {'solver' :('liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga') , 'C':[0.1 , 10]}


# In[150]:


LR = LogisticRegression(n_jobs=-1, random_state=34)


# In[151]:


clf = GridSearchCV(LR, params, cv=5)


# In[152]:


clf.fit(X_train, y_train)


# In[153]:


scores_lr = cross_val_score(clf, X_train, y_train, cv=5)


# In[154]:


scores = clf.cv_results_['mean_test_score']


# In[155]:


clf.best_estimator_


# In[156]:


yhat = clf.predict(X_test)
yhat
yhat_train = clf.predict(X_train)


# In[157]:


yhat_prob = clf.predict_proba(X_test)
yhat_prob
yhat_prob_train = clf.predict_proba(X_train)
yhat_prob_train


# ## Evaluation

# In[158]:


from sklearn.metrics import jaccard_score
print(jaccard_score(y_test, yhat,pos_label=0))
print(jaccard_score(y_train, yhat_train, pos_label = 0))


# In[159]:


from sklearn.metrics import log_loss
print(log_loss(y_test, yhat_prob))
print(log_loss(y_train, yhat_prob_train))


# In[162]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
cnf_matrix=confusion_matrix(y_test, yhat, labels=[1,0])
print(cnf_matrix)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')


# In[163]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)
cnf_matrix


# In[164]:


print (classification_report(y_test, yhat))


# In[165]:


from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

svc = LinearSVC().fit(X_train, y_train)
scores_svc = cross_val_score(svc, X_train, y_train, cv=5)
svc_pred = svc.predict(X_test)

print("Average cross validation score: {:.3f}".format(scores.mean()))
print("Test accuracy: {:.3f}".format(svc.score(X_test, y_test)))
print("F1 score: {:.3f}".format(f1_score(y_test, svc_pred)))
print(confusion_matrix(y_test, svc_pred))


# In[166]:


svc.score(X_train, y_train)


# In[167]:


print (classification_report(y_test, svc_pred))


# In[168]:


svc_pred_train = svc.predict(X_train)
f1_score(y_train, svc_pred_train)