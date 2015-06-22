# Weibo-Suicide-Prediction
This utility which based on jieba, Fast Artificial Neural Network and PyQt4 predicts suicide risk from Chinese Weibo.
<img src="https://raw.githubusercontent.com/LemonChiu/Weibo-Suicide-Prediction/master/screenshots/predict-file.jpg" align="left" width="800">

### Dependency 
+ [jieba](https://github.com/fxsjy/jieba)
+ [Python FANN](https://github.com/FutureLinkCorporation/fann2)
+ [PyQt4](http://www.riverbankcomputing.co.uk/software/pyqt/download/)

This project has been built successfuly under Mac OS X 10.10 Yosemite with jieba 0.36, Python FANN 2.2.0 and PyQt4.

### Usage
1. Run the `mainGUI.py`.
2. Set the number of traning samples from both suicedal and non-suicidal Weibo. You either use the Slider or the SpinBox.
3. Click the `Train` button, it will automatically enter the `train` folder and train from the Weibo files. An input file which fits for the ANN and a trained ANN data will be generated in the `neural-network` folder. Once the ANN system is trained, it can be loaded or executed for many times.
4. Choose the way you provide the Weibo data. If you want to predict many Weibos from files, click the `Predict from file` button and select a file. Please follow the format of `suicide-test.txt` and seperate each element with '\t'. If you just want to test a single Weibo, please fill the Weibo Input form and click the `Predict from input` button.
5. Wait a few seconds and the results will be displayed in the upper Result form.

### License
Licensed under the [MIT](https://github.com/LemonChiu/Weibo-Suicde-Prediction/blob/master/LICENSE) License