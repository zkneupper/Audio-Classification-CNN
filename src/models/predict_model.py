




# Import libraries
import os
import sys

# cpu_count returns the number of CPUs in the system.
from multiprocessing import cpu_count

import numpy as np
import pandas as pd

# Import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Import preprocessing methods from sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

# Import PCA
from sklearn.decomposition import PCA

# Import feature_selection tools
from sklearn.feature_selection import VarianceThreshold

# Import models from sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

# Import XGBClassifier
from xgboost.sklearn import XGBClassifier

# Import from sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

# Import plotting libraries
import matplotlib.pyplot as plt

# Modify notebook settings
pd.options.display.max_columns = 150
pd.options.display.max_rows = 150
%matplotlib inline
plt.style.use('ggplot')


# Create a variable for the project root directory
proj_root = os.path.join(os.pardir)

# Save path to the processed data file
# "dataset_processed.csv"
processed_data_file = os.path.join(proj_root,
                                   "data",
                                   "processed",
                                   "dataset_processed.csv")


# add the 'src' directory as one where we can import modules
src_dir = os.path.join(proj_root, "src")
sys.path.append(src_dir)



# Save the path to the folder that will contain 
# the figures for the final report:
# /reports/figures
figures_dir = os.path.join(proj_root,
                                "reports",
                                "figures")


#Read in the best pipeline

# best_pipeline_file_name = 'pipeline_pickle_20171029.pkl'
best_pipeline_file_name = 'pipeline_pickle.pkl'

best_pipeline_path = os.path.join(models_folder, 
                                  best_pipeline_file_name)

clf = joblib.load(best_pipeline_path)



