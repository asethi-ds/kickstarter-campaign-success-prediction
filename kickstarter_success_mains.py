# Author asethi
# Kickstarter success prediction



#Importing packages
#import researchpy as rp
import numpy as np
import pandas as pd
import os
import glob
from configparser import SafeConfigParser, ConfigParser
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn import metrics
from matplotlib.legend_handler import HandlerLine2D
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
import sys
import pymysql
 
#import source files   
def import_source_files(config_file_name):
    #passing config file information for source file path

    parser = ConfigParser()
    parser.read(config_file_name)

    #Getting source directory path and importing files
    source_dir_path= parser.get('paths', 'source_dir')
    source_dir = os.listdir(source_dir_path)

    if(len(source_dir)):
        all_files = glob.glob(source_dir_path + "/*.csv")
    energy_source_data=pd.read_csv(all_files[0])

    return energy_source_data

   
# Data Preprocessing
def data_preprocess(kickstarter_source_dataset):
 
    kickstarter_source_dataset['state'].value_counts()
 
    # Getting states suspended, cancelled, successful, failed
    kickstarter_projects = kickstarter_source_dataset[(kickstarter_source_dataset['state'] == 'failed')|(kickstarter_source_dataset['state']
    == 'successful')|(kickstarter_source_dataset['state'] == 'canceled')|(kickstarter_source_dataset['state'] == 'suspended')]
 
    # Populating the states canceled, suspended, failed as successful
    kickstarter_projects.loc[kickstarter_projects['state'] == 'canceled', 'state'] = 'unsuccessful'
    kickstarter_projects.loc[kickstarter_projects['state'] == 'suspended', 'state'] = 'unsuccessful'
    kickstarter_projects.loc[kickstarter_projects['state'] == 'failed', 'state'] = 'unsuccessful'
 
    # We now have two states for the target variable, successful or unsuccessful
    kickstarter_projects['state'].value_counts()
 
    ((kickstarter_projects.isnull() | kickstarter_projects.isna()).sum() * 100 / kickstarter_projects.index.size).round(2)
    # We see that 234 missing values in usd pledged, we populate these missing values using values from usd_pledged_real
    kickstarter_projects['usd pledged'].fillna(kickstarter_projects.usd_pledged_real, inplace=True)
    # Removing projects with goal less than 100 dollors
    kickstarter_projects = kickstarter_projects[kickstarter_projects['goal']>100]
 
    return kickstarter_projects
 
#Function for counting syllables in project name
# Semantic
# If first word vowel add one syllable
# Increase counter if vowel is followed by a consonant
# Set minimum syllable value as one.
def syllable_count(project_name):
 
    word=str(project_name).lower()
    count=0
    vowels='aeiou'
    word=str(word)
    first=word[:1]
    if first in vowels:
        count+=1
 
    for index in range(1,len(word)):
 
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count += 1
    return count
 
                                                                                                                                                 # Feature engineering
def feature_engineering(kickstarter_workset):
 
    # Invoking function to calculate syllables in the project name
    kickstarter_workset["syllable_count"]   = kickstarter_workset["name"].apply(lambda x: syllable_count(x))
 
    #Converting launched and deadline values to datetime pandas objects
    kickstarter_workset['launched']         = pd.to_datetime(kickstarter_workset['launched'])
    kickstarter_workset['deadline']         = pd.to_datetime(kickstarter_workset['deadline'])
 
    # Getting values for launched month, launched week, launched day
    kickstarter_workset["launched_month"]   = kickstarter_workset["launched"].dt.month
    kickstarter_workset["launched_week"]    = kickstarter_workset["launched"].dt.week
    kickstarter_workset["launched_day"]     = kickstarter_workset["launched"].dt.weekday
    kickstarter_workset['launched_year']    = kickstarter_workset['launched'].dt.year
    kickstarter_workset['launched_quarter'] = kickstarter_workset['launched'].dt.quarter
 
    # Marking the flag if the launch day falls on the weekend
    kickstarter_workset["is_weekend"]       = kickstarter_workset["launched_day"].apply(lambda x: 1 if x > 4 else 0)
    # Number of words in the name of the projects
    kickstarter_workset["num_words"]        = kickstarter_workset["name"].apply(lambda x: len(str(x).split()))
    # Duration calculation using the differnece between launching date and deadline
    kickstarter_workset["duration"]         = kickstarter_workset["deadline"] - kickstarter_workset["launched"]
    kickstarter_workset["duration"]         = kickstarter_workset["duration"].apply(lambda x: int(str(x).split()[0]))
   
      # Competition evaluation
    # This variable calculates the number of projects launched in the same week belonging to the same category
    kickstarter_workset['launched_year_week_category']        = kickstarter_workset['launched_year'].astype(str)+"_"+kickstarter_workset['launched_week'].astype(str)+"_    "+kickstarter_workset['main_category'].astype(str)
    kickstarter_workset['launched_year_week']                 = kickstarter_workset['launched_year'].astype(str)+"_"+kickstarter_workset['launched_week'].astype(str)
 
    # Getting average number of projects launched per week for each of the main categories category
    kickstarter_workset['week_count']                         = kickstarter_workset.groupby('main_category')['launched_year_week'].transform('nunique')
    kickstarter_workset['project_count_category']             = kickstarter_workset.groupby('main_category')['ID'].transform('count')
    kickstarter_workset['weekly_average_category']            = kickstarter_workset['project_count_category']/kickstarter_workset['week_count']
    kickstarter_workset_category_week=kickstarter_workset[['main_category','weekly_average_category']].drop_duplicates()
 
    #Calculating number of projects launched for a combination of (year,week) for each main category
    kickstarter_workset['weekly_category_launch_count']       = kickstarter_workset.groupby('launched_year_week_category')['ID'].transform('count')
 
    #Competiton quotient
    kickstarter_workset['competition_quotient']               = kickstarter_workset['weekly_category_launch_count']/kickstarter_workset['weekly_average_category']
 
    # Goal Level
    # In this feature we compare the project goal with the mean goal for the category it belongs, the mean goal for the category is used as the normaliation coefficient
    kickstarter_workset['mean_category_goal']                 = kickstarter_workset.groupby('main_category')['goal'].transform('mean')
    kickstarter_workset['goal_level']                         = kickstarter_workset['goal']   / kickstarter_workset['mean_category_goal']
 
    #Duration Level
    # In this feature we compare the project duration with the mean duration for the category with the duration of the given project
    kickstarter_workset['mean_category_duration']             = kickstarter_workset.groupby('main_category')['duration'].transform('mean')
    kickstarter_workset['duration_level']                     = kickstarter_workset['duration']   / kickstarter_workset['mean_category_duration']
 
   
    #Binning the Competition Quotient
    bins_comp_quot                                     = np.array([0,0.25,1,1.5,2.5,10])
    kickstarter_workset["competition_quotient_bucket"] = pd.cut(kickstarter_workset.competition_quotient, bins_comp_quot)
 
    #Binning the Duration Level
    bins_duration_level                                = np.array([0,0.25,1,2,4])
    kickstarter_workset["duration_level_bucket"]       = pd.cut(kickstarter_workset.duration_level, bins_duration_level)                          
    # Binning the USD Goal level
    bins_goal_level                                    = np.array([0,0.5,1.5,5,200])
    kickstarter_workset['goal_level_bucket']           = pd.cut(kickstarter_workset.goal_level, bins_goal_level)                                  
    # Calculating the average amount spent per backer
 
    kickstarter_workset['average_amount_per_backer']   = kickstarter_workset['usd pledged']/kickstarter_workset['backers']
 
    # Marking currency as dollor, euro, gbp and others , this variable is strongly correlated with the country of launch
    kickstarter_workset['currency_usd_flag']   = np.where(kickstarter_workset['currency'] == 'USD',1,0)
 
    # Discarding some features that were created in the intermediate steps and retaining the remaining features
    kickstarter_workset=kickstarter_workset[['ID', 'name', 'category', 'main_category', 'currency', 'deadline','goal',                                     'launched', 'pledged', 'state','backers', 'country',  'usd pledged', 'syllable_count', 'launched_month',
        'launched_day', 'launched_year','launched_quarter', 'is_weekend','num_words',
        'duration','competition_quotient','goal_level', 'duration_level', 'competition_quotient_bucket',
        'duration_level_bucket', 'goal_level_bucket',                                                                                                      'average_amount_per_backer','currency_usd_flag']]
 
   # kickstarter_workset=kickstarter_workset.columns.str.replace(' ','_')
 
    return kickstarter_workset



def pre_model_process(kickstarter_workset):
    
    feature_categorical = ['main_category', 'launched_year', 'launched_quarter','is_weekend',
            'goal_level_bucket','duration_level_bucket']

    feature_numeric = ['average_amount_per_backer', 'goal_level', 'competition_quotient', 'syllable_count','backers']
  
    features_main=feature_categorical+feature_numeric

    for col in feature_categorical:
            kickstarter_workset[col] = kickstarter_workset[col].astype('category')


    kick_projects_ip = pd.get_dummies(kickstarter_workset[features_main],columns = feature_categorical)

    kick_projects_ip = kick_projects_ip.loc[:,~kick_projects_ip.columns.duplicated()]

    kick_projects_ip['state']=kickstarter_workset['state']

    kick_projects_ip=kick_projects_ip.dropna()
    kick_projects_ip=kick_projects_ip[~kick_projects_ip.isin([np.nan, np.inf, -np.inf]).any(1)]

    #kickstarter_workset= kick_projects_ip.dropna()
    #print(kick_projects_ip.isnull().values.any())
    #kick_projects_ip=kick_projects_ip.sample(40000)
    #print('dataset sampled')
    codes = {'successful':0, 'unsuccessful':1}
    kick_projects_ip['state'] = kick_projects_ip['state'].map(codes)
    #kick_projects_ip['state'] = pd.to_numeric(kick_projects_ip['state'], errors='coerce')

    y=kick_projects_ip['state']
    y = pd.DataFrame(y,columns = ['state'])
    colnames = kick_projects_ip.columns
    X=kick_projects_ip[colnames]
    X=X.drop('state', 1)


    X = SelectKBest(f_classif, k=45).fit_transform(X, y)

    #print(X.shape)
    #print(y.shape)
    X_train_out, X_test_out, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=108)

    X_train_out=X_train_out.round(2)
    X_test_out=X_test_out.round(2)

    scaler  = StandardScaler()
    x_train = scaler.fit_transform(X_train_out)
    x_test  = scaler.fit_transform(X_test_out)

    return x_train, x_test, y_train, y_test,colnames


def logistic_reg (x_train, x_test, y_train, y_test):

    logisticRegr = LogisticRegression()
    logisticRegr.fit(x_train, y_train)
    pred_out = logisticRegr.predict(x_test)
    model_accuracy_logistic=accuracy_score(y_test, pred_out)

    #print(confusion_matrix(Y_test_out, pred_out))


    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_out, pos_label=1)
    auc_logistic = auc(fpr, tpr)
    #print(roc_auc)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw,label="area under curve = %1.2f" % auc_logistic)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    #pred_out
    #x_test
    #predicted_results = np.concatenate((x_test, pred_out), axis=1)
    
    return model_accuracy_logistic,auc_logistic
    


def decision_tree (x_train, x_test, y_train, y_test):
    
    clf = DecisionTreeClassifier()
    clf = clf.fit(x_train,y_train)
    

    #Predict the response for train dataset
    pred_out_train = clf.predict(x_train)
    accuracy_dtree = accuracy_score(y_train, pred_out_train)

    #print(confusion_matrix(y_train, pred_out_train))

    #Predict the response for test dataset
    pred_out = clf.predict(x_test)
    #print(accuracy_score(y_test, pred_out))
    #print(confusion_matrix(y_test, pred_out))



    # Since the baseline model is without pruning we get the max depth as 33
    #print(clf.tree_.max_depth)

    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_out, pos_label=1)
    auc_dtree = auc(fpr, tpr)
    #print(roc_auc)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw,label="area under curve = %1.2f" % auc_dtree)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    return accuracy_dtree,auc_dtree


def tree_boosting(x_train, x_test, y_train, y_test):

    clf = GradientBoostingClassifier()

    #training the tree on default features
    clf = clf.fit(x_train,y_train)


    #Predict the response for train dataset
    pred_out_gb = clf.predict(x_train)
    #print(accuracy_score(y_train, pred_out_gb))
    #print(confusion_matrix(y_train, pred_out_gb))


    # Predict the response for test set
    pred_out_gb_test = clf.predict(x_test)
    accuracy_boosting=accuracy_score(y_test, pred_out_gb_test)
    #print(accuracy_score(y_test, pred_out_gb_test))
    #print(confusion_matrix(y_test, pred_out_gb_test))

    # Since the baseline model is without pruning we get the max depth as 33
    #print(clf.tree_.max_depth)

    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_out_gb_test, pos_label=1)
    auc_boosting = auc(fpr, tpr)
    #print(roc_auc)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw,label="area under curve = %1.2f" % auc_boosting)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    return accuracy_boosting, auc_boosting


def cv_logistic(x_train, x_test, y_train, y_test):
    
    clf = LogisticRegression()
    grid_values = {'penalty': ['l1', 'l2']}
    #grid_values = {'penalty': ['l1', 'l2'],'C':[0.001,0.005,0.01,0.5]}
    grid_clf_acc = GridSearchCV(clf, param_grid = grid_values,scoring = 'roc_auc',cv=2)
    grid_clf_acc.fit(x_train,y_train)


    #Predict values based on new parameters
    y_pred_acc     =  grid_clf_acc.predict(x_test)
    acc_logistic_cv   =  accuracy_score(y_test, y_pred_acc)
        
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_acc, pos_label=1)
    roc_auc_cv = auc(fpr, tpr)
    print(roc_auc_cv)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw,label="area under curve = %1.2f" % roc_auc_cv)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    #predicted_results = pd.concat([pd.DataFrame(x_test), y_pred_acc], axis=1)
    return acc_logistic_cv, roc_auc_cv, y_pred_acc



def file_post_processing(config_file_name,x_test,pred_out,colnames_mains):
    
        
    test_mains=pd.DataFrame(x_test)
    pred_mains=pd.DataFrame(pred_out)
    colnames_mains = colnames_mains[:-1]
    test_mains.columns=colnames_mains
    output_set = pd.concat([test_mains,pred_mains],axis=1)
 
    parser = ConfigParser()
    parser.read(config_file_name)
 
    source_dir_path= parser.get('paths', 'source_dir')
    source_dir = os.listdir(source_dir_path )
 
    currentDT = dt.datetime.now()
    name_stamp=currentDT.strftime("%Y-%m-%d")
    processed_dir_path= parser.get('paths','processed_dir')
    processed_dir = os.listdir(processed_dir_path)
    processed_file='processed_data'+str(name_stamp)
 
    file_name = 'processed_data-'+str(dt.datetime.now().strftime("%Y-%m-%d"))
    output_set.to_csv(os.path.join(processed_dir_path,file_name)+'.csv')
 
    #filelist = [ f for f in os.listdir(source_dir_path) if f.endswith(".csv") ]
 
    all_files = glob.glob(source_dir_path + "/*.csv")
 
    for f in all_files:
        os.remove(f)
    
    return output_set


def extract_database_params(config_file_name):
 
    config          = ConfigParser()
    config.read(config_file_name)
 
    db_params=dict()
 
    db_params['db_host']         = config.get('database', 'db_host')
    db_params['db_port']         = config.get('database', 'db_port')
    db_params['db_user']         = config.get('database', 'db_user')
    db_params['db_pass']         = config.get('database', 'db_pass')
    db_params['db_name']         = config.get('database', 'db_name')
 
    return db_params
 
 
  
def append_to_db(database_param_map,df_out):
    # db credentials
    db_host = database_param_map['db_host']
    db_port = int(database_param_map['db_port'])
    db_user = database_param_map['db_user']
    db_pass = database_param_map['db_pass']
    db_name = database_param_map['db_name']
 
    # create connection
    connection = pymysql.connect(host=db_host, user=db_user, port=db_port,
                                     passwd=db_pass, db=db_name)
    cursor = connection.cursor(pymysql.cursors.DictCursor)
 
    # create engine
    engine = create_engine('mysql+pymysql://{a}:{b}@{c}/{d}'.format(a=db_user, b=db_pass, c=db_host, d=db_name))
 
    #print(extract_info.shap9)
 
 
    df_out.to_sql('success_prediction', con=engine, if_exists='append')
    return 1



