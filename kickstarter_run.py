import time
import pandas as pd
start_time = time.time()

import kickstarter_success_mains as ks

#console for global variables and functions call
config_file_name = 'loc_config.ini'
print(config_file_name)
# Getting the two source datasets
# Encoding ISO-8859-1 used since some of the project names have non ascii characters
all_source_files=ks.import_source_files(config_file_name)

kickstarter_source_dataset=pd.read_csv(all_source_files[0], encoding='ISO-8859-1')

kickstarter_workset=ks.data_preprocess(kickstarter_source_dataset)

kickstarter_workset=ks.feature_engineering(kickstarter_workset)
#kickstarter_workset=ks.prepare_model_data(kickstarter_workset)

# Function hypothesis test To be added later
# In this section we carry out hypothesis tests to validate/invalidate some of the assumptions we validate them before we model them
# Test-1 Duration has effect on the state
# Test-2 Length of the project name (syallables) has an effect on the state
# Test-3 Competition has an effect on the state
# Test-4 Quarter and Day of launch effect the state

# Modelling campaign success
#kickstarter_workset.to_csv('kickstarter_head.csv')

# Model data prepare
# Identify  categorical variables, do one hot encoding for ones it is needed
# Identify numeric to be added on the model'
# Split test-train keeping the class balance constant

# Make random forest model
# Get output with metrics

# Make xgboost model
# Get output with metrics

# Ending Print outputs

print("--- %s seconds ---" % (time.time() - start_time))
