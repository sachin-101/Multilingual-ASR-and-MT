# Process and save Common Voice dataset
import re
import os
import string
import pandas as pd
from shutil import copyfile
from pydub import AudioSegment


chars = ['_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', \
         'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
tokens = [i for i in range(len(chars))]
tokenize_dict = {c:t for c,t in zip(chars, tokens)}



def process_sent(s):
    s = s.lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    return ','.join([tokenize_dict[c] for c in s])



def parse_df(df, lan, lang_dir, save_dir):
    """
        Extracts audio and sentence from df
        Returns new df with columns
            "clip" (indexed names)
            "sentence" (corresponding target sentence)
    """
    data = []
    clips_dir = os.path.join(lang_dir, 'clips')
    i = 0
    for clip, sent in zip(df.path, df.sentence):
        try:    # some audio files may not exist
            nums = re.findall(r'\d+', sent)
            if len(nums)==0: # avoiding numbers from target sentences
                clip_name = f'{lan}_{i}.mp3'
                src = os.path.join(clips_dir, clip)
                dst = os.path.join(save_dir, clip_name)
                copyfile(src, dst)
                data.append((clip_name, process_sent(sent)))
                i += 1 # update counter
        except FileNotFoundError:
            pass 
        except TypeError:
            pass
        except KeyError: # for characters such as ú
            pass
    data_df = pd.DataFrame(data, columns=['clip', 'sentence'])
    return data_df


    
if __name__ == "__main__":
    lang_dir = os.path.join('English', 'clips')
    train_dir = 'train_data'
    test_dir = 'test_data'

    # load dataframes
    train_df = pd.read_csv(os.path.join(lang_dir, 'train.tsv'),  delimiter='\t')
    dev_df = pd.read_csv(os.path.join(lang_dir, 'dev.tsv'),  delimiter='\t')
    train_df = pd.concat([train_df, dev_df])
    test_df = pd.read_csv(os.path.join(lang_dir, 'test.tsv'),  delimiter='\t')

    # Extract train data
    print('Preparing Train dataset')
    train_data_df = parse_df(train_df, 'eng', lang_dir, train_dir) # extract data from train_df 
    train_data_df.to_csv(os.path.join(train_dir, 'train_data.csv'))

    # Extract test data
    print('Preparing test dataset')
    test_data_df = parse_df(test_df, 'eng', lang_dir, test_dir) # extract data from test_df
    test_data_df.to_csv(os.path.join(test_dir, 'test_data.csv')) # save