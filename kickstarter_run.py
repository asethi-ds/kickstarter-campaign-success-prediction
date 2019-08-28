#console for global variables and functions call
config_file_name = 'loc_config.ini'

# Getting the two source datasets
# Encoding ISO-8859-1 used since some of the project names have non ascii characters
all_source_files=import_source_files(config_file_name)
kickstarter_source_dataset=pd.read_csv(all_source_files[0], encoding='ISO-8859-1')

kickstarter_workset=data_preprocess(kickstarter_source_dataset)
kickstarter_workset=feature_engineering(kickstarter_workset)
kickstarter_workset=prepare_model_data(kickstarter_workset)

# Hypothesis Test - To be added later
# In this section we carry out hypothesis tests to validate/invalidate some of the assumptions we validate them before we model them
# Test-1 Duration has effect on the state
# Test-2 Length of the project name (syallables) has an effect on the state
# Test-3 Competition has an effect on the state
# Test-4 Quarter and Day of launch effect the state


# Modelling campaign success
kickstarter_workset.to_csv('kickstarter_head.csv')

print("--- %s seconds ---" % (time.time() - start_time))
