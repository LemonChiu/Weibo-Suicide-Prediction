#  encoding=utf-8
__author__ = 'LemonC'

import os
import sys
import timeit
import train_prediction
from fann2 import libfann
from PyQt4 import QtCore, QtGui, uic

# Load .ui file
qtCreatorFile = "mainwindow.ui"

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)


class WeiboApp(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.init_ui()

    def init_ui(self):
        self.center()

        self.actionTrain.triggered.connect(self.train_button_clicked)
        self.actionPredict.triggered.connect(self.predict_file_button_clicked)
        self.actionAbout.triggered.connect(self.about)
        self.actionExit.triggered.connect(QtGui.qApp.quit)

        self.horizontal_slider.valueChanged.connect(self.selected_spinbox.setValue)
        self.selected_spinbox.valueChanged.connect(self.horizontal_slider.setValue)
        self.train_button.clicked.connect(self.train_button_clicked)
        self.predict_file_button.clicked.connect(self.predict_file_button_clicked)
        self.predict_input_button.clicked.connect(self.predict_input_button_clicked)
        self.date_time_edit.setDateTime(QtCore.QDateTime.currentDateTime())

        self.statusBar.showMessage('Ready', 3000)

    def train_button_clicked(self):
        # Loading weibo
        start_time = timeit.default_timer()
        # Get selected Weibo number from spin box
        selected_weibo_num = self.selected_spinbox.value()
        # Select in both suicide and non-suicide weibo (default is 100 * 2 = 200)
        train_weibo_lines = train_prediction.load_train_weibo(selected_weibo_num)
        result_text = 'Loading Weibo files successfully! \n\n'

        # Processing Weibo
        weibo_list = train_prediction.process_weibo(train_weibo_lines)
        stop_time = timeit.default_timer()
        for line in weibo_list:
            result_text += unicode('Content: ' + line[5] + '\nEmotion Grate: ' +
                                   str(line[1]) + '\nSuicide: ' + line[4] + '\n\n')
        result_text += 'Processing Weibo data successfully!\nTotal Weibo processing time: ' + \
                       str('%.2f' % (stop_time - start_time)) + ' seconds.\n\n'

        # Training Weibo
        start_time = timeit.default_timer()
        prediction_accuracy = train_prediction.select_train_set(weibo_list)
        stop_time = timeit.default_timer()
        result_text += 'Training Weibo successfully! Training accuracy is ' + str('%.2f' % prediction_accuracy) + \
                       '%.\nTotal training time: ' + str('%.2f' % (stop_time - start_time)) + ' seconds.\n'
        self.result_text.setText(QtCore.QString(result_text))
        self.result_text.moveCursor(QtGui.QTextCursor.End)
        self.statusBar.showMessage("Training Complete!", 3000)

    def predict_file_button_clicked(self):
        self.statusBar.showMessage("Opening Weibo file...")
        result_text = ''
        filename = QtGui.QFileDialog.getOpenFileName(self, 'Open file', './')
        try:
            with open(filename) as fin_predict:
                start_time = timeit.default_timer()
                weibo_predict_list = train_prediction.process_weibo(fin_predict.readlines()[1:], True)

                os.chdir(os.path.abspath(os.curdir + '/neural-network'))
                ann = libfann.neural_net()
                ann.create_from_file("trained.net")

                # item: [0]id [1]emotion grate [2]time [3]forward [4]content
                for index, item in enumerate(weibo_predict_list):
                    result = ann.run([float(item[1]), float(item[2]), float(item[3])])
                    print('Prediction:' + str('%-18s' % result[0]))
                    prediction = 'No'
                    if (result[0] - 0.5) > 0:
                        prediction = 'Yes'
                    result_text += unicode('Weibo ID: ' + item[0] + '\nContent: ' + item[4] + '\nTime: ' +
                                           str('%.2f' % item[2]) + '(hour)\nForward: ' + item[3] + '\nEmotion Grate: ' +
                                           str(item[1]) + '\nSuicide prediction: ' + prediction + '\n\n')

                stop_time = timeit.default_timer()
                result_text += 'Total prediction time: ' + str('%.2f' % (stop_time - start_time)) + ' seconds.\n'
                self.result_text.setText(QtCore.QString(result_text))
                self.result_text.moveCursor(QtGui.QTextCursor.End)

                # Back to root folder
                os.chdir('..')

                self.statusBar.showMessage('Prediction Complete!', 3000)
        except IOError as err:
            print('Input predict file error: ' + str(err))
            self.statusBar.showMessage("Prediction Fail! Please choose a correct file.", 3000)

    def predict_input_button_clicked(self):
        fake_id = '10000'
        input_text = str(self.weibo_input_text.toPlainText()).replace('\n', ' ')
        if input_text == '':
            self.statusBar.showMessage('Prediction Fail! Please enter the text.', 3000)
            return
        input_date_time = str(self.date_time_edit.dateTime().toString('yyyy.M.d HH:mm'))
        input_forward = str(self.forward_spinbox.value())

        start_time = timeit.default_timer()
        input_line = fake_id + '\t' + input_text + '\t' + input_date_time + '\t' + input_forward
        input_list = list()
        input_list.append(input_line)
        computed_list = train_prediction.process_weibo(input_list, True)

        os.chdir(os.path.abspath(os.curdir + '/neural-network'))
        ann = libfann.neural_net()
        ann.create_from_file("trained.net")

        # single_item: [0]id [1]emotion grate [2]time [3]forward [4]content
        single_item = computed_list[0]    # Only has the first one
        result = ann.run([float(single_item[1]), float(single_item[2]), float(single_item[3])])
        print('Prediction:' + str('%-18s' % result[0]))
        prediction = 'No'
        if (result[0] - 0.5) > 0:
            prediction = 'Yes'
        result_text = unicode('Content: ' + single_item[4] + '\nTime: ' + str('%.2f' % single_item[2]) +
                              '(hour)\nForward: ' + single_item[3] + '\nEmotion Grate: ' +
                              str(single_item[1]) + '\nSuicide prediction: ' + prediction + '\n\n')

        stop_time = timeit.default_timer()
        result_text += 'Total prediction time: ' + str('%.2f' % (stop_time - start_time)) + ' seconds.\n'
        self.result_text.setText(QtCore.QString(result_text))
        self.result_text.moveCursor(QtGui.QTextCursor.End)

        # Back to root folder
        os.chdir('..')

        self.statusBar.showMessage('Prediction Complete!', 3000)

    # Set window center
    def center(self):
        screen = QtGui.QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2)

    # Overwrite about()
    def about(self):

        QtGui.QMessageBox.about(self, 'About Weibo Prediction',
                                '<h2>Weibo Suicide Prediction 1.0</h2>'
                                '<p>This program uses:<br />'
                                'Jieba 0.36 <br />'
                                'Fast Artificial Neural Networks 2.2.0 <br />'
                                'PyQt4 </p>'
                                '<p>Licenced under MIT</p>')

    # Set shortcut for close()
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    main_window = WeiboApp()
    main_window.show()
    sys.exit(app.exec_())
