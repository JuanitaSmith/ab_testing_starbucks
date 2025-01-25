""" default path and file locations """
import os

# get working directory
project_root = os.getcwd()

# strip file path to the root directory of the project
project_root = project_root.split(sep='/src')[0]
project_root = project_root.split(sep='/tests')[0]
project_root = project_root.split(sep='/notebooks')[0]

# FOLDER STRUCTURE
folder_data = 'data'
folder_raw = 'data/raw'
folder_clean = 'data/clean'
folder_logs = 'logs'
folder_db = 'sqlite:///'
folder_scripts = 'src'
folder_models = 'models'

# LOG NAMES
filename_log_ml = 'train_classifier.log'

# FILE/TABLE NAMES
filename_train_data = 'training.csv'
filename_test_data = 'Test.csv'
# filename_database = '***.db'
filename_database_optuna = 'starbucks.db'
filename_model_prep = 'classifier_prep.pkl'
filename_model = 'classifier.pkl'
filename_optuna_study = 'optuna_study.pkl'
filename_translations = 'translations.csv'

# FILE PATHS TO FULL FILE NAMES
path_log_ml = os.path.join(project_root, folder_logs, filename_log_ml)
path_train_data = os.path.join(project_root, folder_raw, filename_train_data)
path_train_dat_clean = os.path.join(project_root, folder_clean, filename_train_data)
path_test_data = os.path.join(project_root, folder_raw, filename_test_data)
path_database_optuna = (folder_db + project_root + '/' + folder_clean + '/' + filename_database_optuna)
