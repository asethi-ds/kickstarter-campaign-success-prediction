# Kickstarter
# Data cleaning, preprocessing and exploratory analysis
# Author Asethi
# Last updated 8/22/2019

# Importing packages
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


# Console for global variables and functions call
config_file_name = 'loc_config.ini'

# Getting the two source datasets
all_source_files=import_source_files(config_file_name)
kickstarter_source_dataset_one=pd.read_csv(all_source_files[0], encoding='ISO-8859-1')
kickstarter_source_dataset_two=pd.read_csv(all_source_files[1], encoding='ISO-8859-1')
