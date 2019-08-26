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
def data_preprocess(kicksstarter_workset):

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
    
    return kickstarter_projects

#Function for counting syllables in project name
# Semantic
# If first word vowel add one syllable
# Increase counter if vowel is followed by a consonant
# Set minimum syllable value as one.
def syllable_count(project_name):
        
    word=project_name.lower()
    count=0
    vowels='aeiouy'
    
    if word[0] in vowels:
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
    # Getting values for launched month, launched week, launched day
    kickstarter_workset["launched_month"]   = kickstarter_workset["launched"].dt.month
    kickstarter_workset["launched_week"]    = kickstarter_workset["launched"].dt.week
    kickstarter_workset["launched_day"]     = kickstarter_workset["launched"].dt.weekday
    # Marking the flag if the launch day falls on the weekend
    kickstarter_workset["is_weekend"]       = kickstarter_workset["launched_day"].apply(lambda x: 1 if x > 4 else 0)
    # Number of words in the name of the projects
    kickstarter_workset["num_words"]        = kickstarter_workset["name"].apply(lambda x: len(x.split()))
    # Duration calculation using the differnece between launching date and deadline
    kickstarter_workset["duration"]         = kickstarter_workset["deadline"] - kickstarter_workset["launched"]
    kickstarter_workset["duration"]         = kickstarter_workset["duration"].apply(lambda x: int(str(x).split()[0]))
    # Competition evaluation
    # This variable calculates the number of projects launched in the same week belonging to the same category
    #kickstarter_workset['']
    

    return kickstarter_workset


# Console for global variables and functions call
config_file_name = 'loc_config.ini'

# Getting the two source datasets
# Encoding ISO-8859-1 used since some of the project names have non ascii characters
all_source_files=import_source_files(config_file_name)
kickstarter_source_dataset=pd.read_csv(all_source_files[0], encoding='ISO-8859-1')

# 
kickstarter_workset=feature_engineering(kickstarter_source_dataset)

kickstarter_workset=data_preprocess(kickstarter_workset)

print("--- %s seconds ---" % (time.time() - start_time))

