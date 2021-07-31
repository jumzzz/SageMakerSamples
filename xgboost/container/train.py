#usr/bin/python
import os
import argparse

import pandas as pd
import xgboost as xgb


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument()


def get_files(dir_path):
    fnames = os.listdir(dir_path)
    fpaths = [os.path.join(dir_path, fn) for fn in fnames if '.csv' in fn]
    return fpaths


def csvs_to_df(csvs):
    df = pd.concat([pd.read_csv(csv) for csv in csvs])
    return df


def get_df(target_dir):
    fpaths = get_files(target_dir)
    df = csvs_to_df(fpaths)
    return df


def get_feat_target(df):
    
    feat_cols = [ 
        'SepalLengthCm',
        'SepalWidthCm',
        'PetalLengthCm',
        'PetalWidthCm',
    ]

    X = df[feat_cols]
    y = df['target']
    
    return X,y


def dump_model(model):
    target_dir = os.environ['SM_MODEL_DIR']
    target_path = os.path.join(target_dir, 'model.pkl')
    model.save_model(target_path)


def train():
    
    train_dir = os.environ['SM_CHANNEL_TRAINING'] 
    ts_dir = os.environ['SM_CHANNEL_TESTING']

    df_tr = get_df(train_dir)
    df_ts = get_df(ts_dir)
    
    X_tr, y_tr = get_feat_target(df_tr)
    X_ts, y_ts = get_feat_target(df_ts)
    
    model = xgb.XGBClassifier(objective='multi:softmax', num_round=100)
    model.fit(X_tr, y_tr, eval_set=[(X_ts, y_ts)])
    
    dump_model(model)

 
if __name__ == "__main__":
    train()

