import argparse
import os
import glob
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, default="")
    parser.add_argument("--src_dir_alternative", type=str, default="")
    parser.add_argument(
        "--tgt_dir", type=str, default=""
    )
    args = parser.parse_args()
    return args

def write_list(request, result):
    write_queue.put(result)

def split_dataset(df, total_folds, fold, key="pid"):
    kfold = KFold(total_folds, shuffle=True, random_state=42)
    train_idx, val_idx = list(kfold.split(df['pid'], df['source']))[fold]
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    print("train: {}, val: {}".format(len(train_df), len(val_df)))
    return train_df, val_df

def split_cycle(args):
    total_folds = 10
    save_prefix = args.tgt_dir
    data_list = os.listdir(args.src_dir)
    #data_list = glob.glob(os.path.join(args.tgt_dir, "*.npz"))
    #data_list = [os.path.basename(os.path.splitext(it)[0]) for it in data_list]
    df = pd.DataFrame(data_list, columns=['pid'])
    source_list = [pid.split('-')[0] for pid in data_list]
    df['source'] = pd.DataFrame(source_list, columns=['source'])
    print(f"len(df): {df.head()}")
    for idx, fold in enumerate(range(total_folds)):
        train_df, val_df = split_dataset(df, total_folds, fold)
        train_pids = train_df['pid'].values
        val_pids = val_df['pid'].values
        
        train_pids = [v + '\n' for v in list(set(train_pids))]
        val_pids = [v + '\n' for v in list(set(val_pids))]
        
        with open(save_prefix + f'/train_split_{idx+1}.lst', 'w') as f:
            f.writelines(train_pids)
    
        with open(save_prefix + f'/val_split_{idx+1}.lst', 'w') as f:
            f.writelines(val_pids)

def gen_pretrain_lst(args):
    save_prefix = args.tgt_dir
    anno_list = os.listdir(args.src_dir)
    alternative_list = os.listdir(args.src_dir_alternative)
    data_list = [pid for pid in alternative_list if pid not in anno_list]
    print(f"anno_list: {len(anno_list)}, alternative_list: {len(alternative_list)}, data_list: {len(data_list)}")
    pretrain_pids = [v + '\n' for v in list(set(data_list))]
    with open(save_prefix + f'/pretrain.lst', 'w') as f:
        f.writelines(pretrain_pids)
    

if __name__ == "__main__":
    args = parse_args()
    split_cycle(args)
    #gen_pretrain_lst(args)
