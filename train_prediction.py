#  encoding=utf-8
__author__ = 'LemonC'

import os
import sys
import jieba
import random
import timeit
from fann2 import libfann

reload(sys)
sys.setdefaultencoding('utf-8')


# Load weibo and parameter is the specified weibo lines each file
def load_train_weibo(selected_count):
    train_lines = []
    # /Users/LemonC/Code/Python/WeiboPrediction/train
    os.chdir(os.path.abspath(os.curdir + '/train'))
    # Read suicide file
    try:
        with open('suicide.txt') as fin_suicide:
            # Ignore the first line
            train_lines = random.sample(fin_suicide.readlines()[1:], selected_count)
            print('Load suicide file (' + str(selected_count) + ' weibo).')
    except IOError as err:
        print('Suicide file input error: ' + str(err))
    # Read non suicide filer
    try:
        with open('nonsuicide.txt') as fin_not_suicide:
            # Ignore the first line and integrate to the trainLines
            train_lines += random.sample(fin_not_suicide.readlines()[1:], selected_count)
            print('Load non-suicide file (' + str(selected_count) + ' weibo).')
    except IOError as err:
        print('Non-Suicide file input error: ' + str(err))

    # Back to root folder
    os.chdir('..')

    return train_lines


# Process raw Weibo data. If parameter "predict" is False, read the suicide label.
# Return a Weibo list with emotion grate
def process_weibo(all_weibo_lines, predict=False):
    start_time = timeit.default_timer()

    # Parameter is the parallel process number, NOT support Windows
    jieba.enable_parallel(4)
    # Add user dict and word
    jieba.add_word('不像')

    not_dict = []
    degree_dict = {}
    negative_dict = []
    positive_dict = []

    """
    Load Dictionaries
    """
    print('Current path: ' + os.path.abspath(os.curdir) + '\n')
    # /Users/LemonC/Code/Python/WeiboPrediction/dictionary
    os.chdir(os.path.abspath(os.curdir + '/dictionary'))
    # Not dictionary
    try:
        with open('not-dict.txt') as fin_not_dict:
            for line in fin_not_dict:
                not_dict.append(line.strip())
    except IOError as err:
        print('Not Dictionary input error: ' + str(err))
    # Degree dictionary
    try:
        with open('degree.txt') as fin_degree_dict:
            for line in fin_degree_dict:
                words = line.strip().split(' ')
                degree_dict[words[0]] = words[1]
            # Define weight
            for item in degree_dict:
                if degree_dict[item] == '1':    # 1最|most 2
                    degree_dict[item] = 2
                if degree_dict[item] == '2':    # 2很|very 1.25
                    degree_dict[item] = 1.25
                if degree_dict[item] == '3':    # 3较|more 1.2
                    degree_dict[item] = 1.2
                if degree_dict[item] == '4':    # 4稍|-ish 0.8
                    degree_dict[item] = 0.8
                if degree_dict[item] == '5':    # 5欠|insufficiently 0.5
                    degree_dict[item] = 0.5
                if degree_dict[item] == '6':    # 6超|over 1.5
                    degree_dict[item] = 1.5
    except IOError as err:
        print('Degree Dictionary input error: ' + str(err))
    # NTUSD negative dictionary
    try:
        with open('ntusd-negative.txt') as fin_negative:
            for line in fin_negative:
                negative_dict.append(line.strip())
    except IOError as err:
        print('Negative Dictionary input error: ' + str(err))
    # Supplement (negative) Dictionary
    try:
        with open('supplement.txt') as fin_supplement:
            negative_supplement = []
            for line in fin_supplement:
                negative_supplement.append(line.strip())
            # Integrate negative supplement including negative expression
            negative_dict += negative_supplement
    except IOError as err:
        print('Positive Dictionary input error: ' + str(err))
    # NTUSD positive dictionary
    try:
        with open('ntusd-positive.txt') as fin_positive:
            for line in fin_positive:
                positive_dict.append(line.strip())
    except IOError as err:
        print('Positive Dictionary input error: ' + str(err))

    """
    Process Weibo
    """
    weibo_items = []
    for line in all_weibo_lines:
        weibo_item = []
        # [0]id [1]content [2]date [3]type [4]suicide
        item_elements = line.split('\t')
        weibo_item.append(item_elements[0])   # id
        """
        Compute Emotion Grate
        """
        weibo_content = item_elements[1]
        emotion_grade = 0
        last_emotion_position = 0

        # Default mode
        seg_list = jieba.cut(weibo_content)
        content_words = list(seg_list)
        print(" ".join(content_words))

        for i in range(0, len(content_words)):
            if (content_words[i] in negative_dict) or (content_words[i] in positive_dict):
                # Initialize parameters
                remedy_weight = 1
                degree_weight = 1
                degree_position = 0
                not_position = 0
                if content_words[i] in negative_dict:
                    emotion_weight = -1
                else:
                    emotion_weight = 1

                # Between two emotion words
                for j in range(last_emotion_position, i):
                    # Get REMEDY weight
                    if content_words[j] in not_dict:
                        not_position = j
                        # Avoid too far.
                        if (i - j) < 8:
                            remedy_weight = -1

                    # Get DEGREE weight
                    for key, value in degree_dict.iteritems():
                        if content_words[j] == key:
                            degree_position = j
                            degree_weight = value

                    # Both NOT and DEGREE exist
                    if (not_position * degree_position != 0) and (not_position - degree_position):
                        remedy_weight = -0.5

                last_emotion_position = i
                emotion_grade += remedy_weight * degree_weight * emotion_weight
        print('Emotion Grade: ' + str(emotion_grade))
        weibo_item.append(emotion_grade)    # emotion grade

        """
        Compute time
        """
        # Ignore date
        weibo_time = item_elements[2].split(' ')[1]
        # Convert to hour
        weibo_time = float(weibo_time.split(':')[0]) + float(weibo_time.split(':')[1]) / 60
        weibo_item.append(weibo_time)    # Time
        weibo_item.append(item_elements[3])   # Forward
        # It has a suicide label
        if predict is False:
            weibo_item.append(item_elements[4].strip())   # Suicide or not
        weibo_item.append(item_elements[1])    # Origin content

        # WeiboItem [0]id [1]emotion grate [2]time [3]forward [4]suicide or not [5]content
        weibo_items.append(weibo_item)

    stop_time = timeit.default_timer()
    print('Weibo Processing time: ' + str('%.2f' % (stop_time - start_time) + ' seconds.\n'))
    # Back to root folder
    os.chdir('..')
    return weibo_items


# Randomly select train set and train neural network
def select_train_set(weibo_items):
    start_time = timeit.default_timer()
    """
    Randomly Select Train Set
    """
    # 75% total
    train_list = random.sample(weibo_items, len(weibo_items) * 3 / 4)
    # 25% for test
    test_list = []
    for item in weibo_items:
        if item not in train_list:
            test_list.append(item)

    train_set_lines = ''
    # First line: #pairs #inputs #output
    train_set_lines += str(len(train_list)) + ' 3 1\n'
    # [1]emotion grate [2]time [3]forward
    # [4]suicide or not
    for item in weibo_items:
        train_set_lines += str(item[1]) + ' ' + str(item[2]) + ' ' + str(item[3]) + ' \n'
        train_set_lines += str(item[4]) + '\n'

    # Create neural network folder
    if not(os.path.exists('neural-network')):
        os.mkdir('neural-network')
    # /Users/LemonC/Code/Python/WeiboPrediction/neural-network
    os.chdir(os.path.abspath(os.curdir + '/neural-network'))
    # Write input file for neural network
    with open('NN-train-set.txt', 'w') as fout_for_NN:
        fout_for_NN.write(train_set_lines)
        print('Prepare train set for Neural Network successfully')

    """
    Neural Network Training
    """
    connection_rate = 1    # full connect
    learning_rate = 0.7
    number_input = 3
    number_hidden = 5
    number_output = 1

    desired_error = 0.001
    max_iterations = 10000
    iterations_between_reports = 1000

    ann = libfann.neural_net()
    ann.create_sparse_array(connection_rate, (number_input, number_hidden, number_output))
    ann.set_learning_rate(learning_rate)
    ann.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC)    # (-1, 1)
    ann.set_activation_function_output(libfann.SIGMOID)    # (0,1)

    ann.train_on_file('NN-train-set.txt', max_iterations, iterations_between_reports, desired_error)
    ann.save('trained.net')
    ann.destroy()
    stop_time = timeit.default_timer()
    print('Total training time: ' + str('%.2f' % (stop_time - start_time) + ' seconds.\n'))

    # Back to root folder
    os.chdir('..')
    return test_prediction(test_list)


# Test prediction. Return accuracy
def test_prediction(test_list):
    os.chdir(os.path.abspath(os.curdir + '/neural-network'))
    ann = libfann.neural_net()
    ann.create_from_file("trained.net")
    correct_count = 0

    # item: [0]id [1]emotion grate [2]time [3]forward [4]suicide or not
    for item in test_list:
        result = ann.run([float(item[1]), float(item[2]), float(item[3])])
        print('Prediction:' + str('%-18s' % result[0]) + 'Suicide:' + item[4])

        prediction = 0
        if (result[0] - 0.5) > 0:
            prediction = 1
        if int(item[4]) == prediction:
            correct_count += 1

    accuracy = round(float(correct_count) / len(test_list) * 100, 2)
    print('Results: Correct number is ' + str(correct_count) + '. Model accuracy is ' + str('%.2f' % accuracy) + '%.')

    # Back to root folder
    os.chdir('..')

    return accuracy

if __name__ == '__main__':
    # Select 100 in both suicide and non-suicide weibo (total 200)
    select_num = 100
    train_weibo_lines = load_train_weibo(select_num)
    weibo_list = process_weibo(train_weibo_lines)
    select_train_set(weibo_list)
