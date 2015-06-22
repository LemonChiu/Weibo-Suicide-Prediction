# encoding=utf-8
__author__ = 'LemonC'

import glob
import train_prediction

# Load Weibo file and return a weibo list
def predict(txt_file):
    try:
        with open(txt_file) as fin_weibo:
            print('Loading weibo file: ' + txt_file)
            # Ignore the first line
            return train_prediction.process_weibo(fin_weibo.readlines()[1:])
    except IOError as err:
        print('Weibo files input error: ' + str(err))

if __name__ == '__main__':
    # Search all input weibo(.txt) and execute
    for weibo_file in glob.glob('*.txt'):
        test_list = predict(weibo_file)
        train_prediction.test_prediction(test_list)
