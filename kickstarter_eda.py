# Kickstarter
# Data cleaning, preprocessing and exploratory analysis
# Author asethi
# Last updated 8/22/2019

# Importing packages
import sys
# Clearing the memory
sys.modules[__name__].__dict__.clear()
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
def data_preprocess(kicksstarter_source_dataset):

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
    
    return kickstarter_projects




# Console for global variables and functions call
config_file_name = 'loc_config.ini'

# Getting the two source datasets
# Encoding ISO-8859-1 used since some of the project names have non ascii characters
all_source_files=import_source_files(config_file_name)
kickstarter_source_dataset=pd.read_csv(all_source_files[0], encoding='ISO-8859-1')


kickstarter_workset=data_preprocess(kickstarter_source_dataset)


print("--- %s seconds ---" % (time.time() - start_time))

