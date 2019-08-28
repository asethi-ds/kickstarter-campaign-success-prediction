# Kickstarter
# Data cleaning, preprocessing and exploratory analysis
# Author asethi
# Last updated 8/22/2019

# Importing packages
import sys
# Clearing the memory
#sys.modules[__name__].__dict__.clear()
import time
start_time = time.time()
import pandas as pd
import numpy as np
import datetime as dt
import os
import glob
from configparser import SafeConfigParser, ConfigParser

# Function definitions

#Function to get source data
def import_source_files(config_file_name):
    #passing config file information for source file path
    
    parser = ConfigParser()
    parser.read(config_file_name)

    #Getting source directory path and importing files
    source_dir_path= parser.get('paths', 'source_dir')
    source_dir = os.listdir(source_dir_path)

    if(len(source_dir)):
        all_files = glob.glob(source_dir_path + "/*.csv")
    
    return all_files


# Data Preprocessing
def data_preprocess(kickstarter_workset):

    kickstarter_source_dataset['state'].value_counts()

    # Getting states suspended, cancelled, successful, failed
    kickstarter_projects = kickstarter_workset[(kickstarter_source_dataset['state'] == 'failed')|(kickstarter_source_dataset['state']
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
    kickstarter_workset=kickstarter_workset[['ID', 'name', 'category', 'main_category', 'currency', 'deadline','goal', 'launched', 'pledged', 'state',
        'backers', 'country',  'usd pledged', 'syllable_count', 'launched_month',  'launched_day', 'launched_year','launched_quarter', 'is_weekend','num_words',
        'duration','competition_quotient','goal_level', 'duration_level', 'competition_quotient_bucket','duration_level_bucket', 'goal_level_bucket',
        'average_amount_per_backer','currency_usd_flag']]
   


    kickstarter_workset=kickstarter_workset.columns.str.replace(' ','_')

    return kickstarter_workset


# Console for global variables and functions call
config_file_name = 'loc_config.ini'

# Getting the two source datasets
# Encoding ISO-8859-1 used since some of the project names have non ascii characters
all_source_files=import_source_files(config_file_name)
kickstarter_source_dataset=pd.read_csv(all_source_files[0], encoding='ISO-8859-1')

kickstarter_workset=data_preprocess(kickstarter_source_dataset)
kickstarter_workset=feature_engineering(kickstarter_workset)

# Hypothesis Test - To be added later
# In this section we carry out hypothesis tests to validate/invalidate some of the assumptions we validate them before we model them
# Test-1 Duration has effect on the state
# Test-2 Length of the project name (syallables) has an effect on the state
# Test-3 Competition has an effect on the state
# Test-4 Quarter and Day of launch effect the state


# Modelling campaign success


print("--- %s seconds ---" % (time.time() - start_time))

