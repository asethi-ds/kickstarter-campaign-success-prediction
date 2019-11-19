import sys
import time
import logging
import kickstarter_success_mains as ks
start_time = time.time()
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',level=logging.DEBUG,datefmt='%Y-%m-%d %H:%M:%S',filename='model_run.log')

logging.debug('Model Run at datetime')

try:

    config_file_name                                   = 'loc_config.ini'
    kickstarter_source_data                            = ks.import_source_files(config_file_name)
    logging.debug('Projects source data imported')
    logging.info('%s projects imported with %s columns' % (kickstarter_source_data.shape[0], kickstarter_source_data.shape[1]))
    kickstarter_workset                                = ks.data_preprocess(kickstarter_source_data)
    logging.debug('Data-Preprocessed')
    kickstarter_workset                                = ks.feature_engineering(kickstarter_workset)
    logging.debug('Feature Engineering Completed')
    x_train, x_test, y_train, y_test,colnames_mains    = ks.pre_model_process(kickstarter_workset)
    logging.debug('Data prepared for model implementation')
    accuracy_logistic,auc_logistic                     = ks.logistic_reg(x_train, x_test, y_train, y_test)
    logging.info('logistic regression run with  %s accuracy and % auc' % (accuracy_logistic, auc_logisitc))   
    accuracy_dtree,auc_dtree                           = ks.decision_tree(x_train, x_test, y_train, y_test)
    logging.info('decision tree run with  %s accuracy and % auc' % (accuracy_dtree, auc_dtree))
    accuracy_boosting,auc_boosting                     = ks.tree_boosting(x_train, x_test, y_train, y_test)
    logging.info('boosting model run with  %s accuracy and % auc' % (accuracy_boosting, auc_boosting))
    accuracy_final,auc_final,pred_out                  = ks.cv_logistic(x_train, x_test, y_train, y_test)
    logging.info('final implementation with  %s accuracy and % auc' % (accuracy_final, auc_final))
    output_kickstarter_pred                            = ks.file_post_processing(config_file_name,x_test,pred_out,colnames_mains)
    database_param_map                                 = ks.extract_database_params(config_file_name)
    logging.debug('db params mapping retrieved')
    success                                            = ks.append_to_db(database_param_map,output_kickstarter_pred)
    logging.debug('Success prediction results stored %s') 


except Exception as exc:
    exc_type, exc_obj, exc_tb = sys.exc_info()
                 
    print(exc_type)
    print(exc_type)
    print(exc_obj)
    print(exc_tb.tb_lineno)
    
finally:
    pass
