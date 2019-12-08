# Process and save Common Voice dataset
import os
from pydub import AudioSegment
import pandas as pd
import re

dataset_dir = os.path.join('..','..','Dataset')
cv_dir = os.path.join(dataset_dir, 'Common voice')


def create_dirs(lang):
    lang_dir = os.path.join(cv_dir, lang)
    train_dir = os.path.join(lang_dir, 'train_data')
    test_dir = os.path.join(lang_dir, 'test_data')
    try:
        os.mkdir(train_dir)
    except FileExistsError:
        pass
    try:
        os.mkdir(test_dir)
    except FileExistsError:
        pass
    return lang_dir, train_dir, test_dir


def parse_df(df, lan, lang_dir, save_dir):
    """
        Extracts audio and sentence from df
        Converts mp3 audio to wav
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
                clip_name = f'{lan}_{i}.wav'
                save_clip_dir = os.path.join(save_dir, clip_name)
                convert_to_wav(clips_dir, clip, save_clip_dir)
                data.append((clip_name, sent))
                i += 1 # update counter
        except FileNotFoundError:
            pass
        except TypeError:
            pass
    data_df = pd.DataFrame(data, columns=['clip', 'sentence'])
    return data_df
        

def convert_to_wav(clips_dir, clip, save_clip_dir):
    """
        Converts and saves mp3 to wav
    """
    mp3_dir = os.path.join(clips_dir, clip)
    mp3_file = AudioSegment.from_mp3(mp3_dir)
    mp3_file.export(save_clip_dir, format='wav')

    
if __name__ == "__main__":
    lang = "English"
    lang_dir, train_dir, test_dir = create_dirs(lang)

    # load dataframes
    train_df = pd.read_csv(os.path.join(lang_dir, 'train.tsv'),  delimiter='\t')
    dev_df = pd.read_csv(os.path.join(lang_dir, 'dev.tsv'),  delimiter='\t')
    test_df = pd.read_csv(os.path.join(lang_dir, 'test.tsv'),  delimiter='\t')

    # Extract train data
    train_data_df = parse_df(train_df, 'eng', lang_dir, train_dir) # extract data from train_df 
    dev_data_df = parse_df(dev_df, 'eng', lang_dir, train_dir) # extract data from train_df
    total_train_data = pd.concat([train_data_df, dev_data_df])
    total_train_data.to_csv(os.path.join(train_dir, 'train_data.csv')) # save
    
    # Extract test data
    test_data_df = parse_df(test_df, 'eng', lang_dir, test_dir) # extract data from test_df
    test_data_df.to_csv(os.path.join(test_dir, 'test_data.csv')) # save