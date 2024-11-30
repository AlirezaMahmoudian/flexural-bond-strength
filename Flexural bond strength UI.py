#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PyQt5.QtGui import QFont
from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1210, 692)
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(240, 70, 191, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(13)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(250, 120, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(13)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(250, 220, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(13)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setGeometry(QtCore.QRect(250, 270, 271, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(13)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(Form)
        self.label_5.setGeometry(QtCore.QRect(250, 320, 371, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(13)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_9 = QtWidgets.QLabel(Form)
        self.label_9.setGeometry(QtCore.QRect(270, 570, 321, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(15)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.lineEdit = QtWidgets.QLineEdit(Form)
        self.lineEdit.setGeometry(QtCore.QRect(610, 270, 121, 31))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(Form)
        self.lineEdit_2.setGeometry(QtCore.QRect(610, 320, 121, 31))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_3 = QtWidgets.QLineEdit(Form)
        self.lineEdit_3.setGeometry(QtCore.QRect(610, 120, 121, 31))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.label_11 = QtWidgets.QLabel(Form)
        self.label_11.setGeometry(QtCore.QRect(10, 20, 181, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(13)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.label_13 = QtWidgets.QLabel(Form)
        self.label_13.setGeometry(QtCore.QRect(250, 170, 241, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(13)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.lineEdit_4 = QtWidgets.QLineEdit(Form)
        self.lineEdit_4.setGeometry(QtCore.QRect(610, 220, 121, 31))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.radioButton = QtWidgets.QRadioButton(Form)
        self.radioButton.setGeometry(QtCore.QRect(20, 90, 91, 20))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.radioButton.setFont(font)
        self.radioButton.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(Form)
        self.radioButton_2.setGeometry(QtCore.QRect(20, 160, 91, 20))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.radioButton_2.setFont(font)
        self.radioButton_2.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.radioButton_2.setObjectName("radioButton_2")
        self.radioButton_3 = QtWidgets.QRadioButton(Form)
        self.radioButton_3.setGeometry(QtCore.QRect(20, 230, 91, 20))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.radioButton_3.setFont(font)
        self.radioButton_3.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.radioButton_3.setObjectName("radioButton_3")
        self.radioButton_4 = QtWidgets.QRadioButton(Form)
        self.radioButton_4.setGeometry(QtCore.QRect(20, 300, 91, 20))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.radioButton_4.setFont(font)
        self.radioButton_4.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.radioButton_4.setObjectName("radioButton_4")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(420, 520, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.predict)
        self.label_6 = QtWidgets.QLabel(Form)
        self.label_6.setGeometry(QtCore.QRect(250, 370, 221, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(13)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(Form)
        self.label_7.setGeometry(QtCore.QRect(250, 410, 151, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(13)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.lineEdit_9 = QtWidgets.QLineEdit(Form)
        self.lineEdit_9.setGeometry(QtCore.QRect(610, 560, 341, 51))
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.line = QtWidgets.QFrame(Form)
        self.line.setGeometry(QtCore.QRect(0, 490, 1231, 16))
        self.line.setLineWidth(3)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.comboBox = QtWidgets.QComboBox(Form)
        self.comboBox.setGeometry(QtCore.QRect(610, 70, 121, 31))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox_2 = QtWidgets.QComboBox(Form)
        self.comboBox_2.setGeometry(QtCore.QRect(610, 170, 121, 31))
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_3 = QtWidgets.QComboBox(Form)
        self.comboBox_3.setGeometry(QtCore.QRect(610, 360, 121, 31))
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_4 = QtWidgets.QComboBox(Form)
        self.comboBox_4.setGeometry(QtCore.QRect(610, 410, 121, 31))
        self.comboBox_4.setObjectName("comboBox_4")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.lineEdit_10 = QtWidgets.QLineEdit(Form)
        self.lineEdit_10.setGeometry(QtCore.QRect(610, 630, 341, 51))
        self.lineEdit_10.setObjectName("lineEdit_10")
        self.label_10 = QtWidgets.QLabel(Form)
        self.label_10.setGeometry(QtCore.QRect(270, 640, 321, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(15)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.label_12 = QtWidgets.QLabel(Form)
        self.label_12.setGeometry(QtCore.QRect(780, 80, 411, 351))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(15)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_12.setFont(font)
        self.label_12.setText("")
        self.label_12.setPixmap(QtGui.QPixmap("C:/Users/ALIREZA/Desktop/Picture1.png.tif"))
        self.label_12.setScaledContents(True)
        self.label_12.setObjectName("label_12")
        self.label_14 = QtWidgets.QLabel(Form)
        self.label_14.setGeometry(QtCore.QRect(240, 20, 181, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(13)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "ML models UI"))
        self.label.setText(_translate("Form", "Type of concrete:"))
        self.label_2.setText(_translate("Form", "fc (MPa):"))
        self.label_3.setText(_translate("Form", "Bar size (mm):"))
        self.label_4.setText(_translate("Form", "Cmin (mm):"))
        self.label_5.setText(_translate("Form", "Embedment lengths (mm):"))
        self.label_9.setText(_translate("Form", "τmax (MPa):"))
        self.label_11.setText(_translate("Form", "Model selection:"))
        self.label_13.setText(_translate("Form", "Type of bar:"))
        self.radioButton.setText(_translate("Form", "DT     "))
        self.radioButton_2.setText(_translate("Form", "RF     "))
        self.radioButton_3.setText(_translate("Form", " GB     "))
        self.radioButton_4.setText(_translate("Form", "XGB  "))
        self.pushButton_2.setText(_translate("Form", "Prediction"))
        self.label_6.setText(_translate("Form", "Type of Condition:"))
        self.label_7.setText(_translate("Form", "Contaminant:"))
        self.comboBox.setItemText(0, _translate("Form", "NC"))
        self.comboBox.setItemText(1, _translate("Form", "SCC\n"))
        self.comboBox.setItemText(2, _translate("Form", "FRC\n"))
        self.comboBox.setItemText(3, _translate("Form", "ECC\n\n"))
        self.comboBox.setItemText(4, _translate("Form", "NC-HIGH\n"))
        self.comboBox.setItemText(5, _translate("Form", "HPC\n"))
        self.comboBox.setItemText(6, _translate("Form", "UHPC"))
        self.comboBox_2.setItemText(0, _translate("Form", "Steel"))
        self.comboBox_2.setItemText(1, _translate("Form", "Sand-coated GFRP"))
        self.comboBox_2.setItemText(2, _translate("Form", "Low moduls GFRP"))
        self.comboBox_2.setItemText(3, _translate("Form", "Helically GFRP"))
        self.comboBox_2.setItemText(4, _translate("Form", "BFRP"))
        self.comboBox_2.setItemText(5, _translate("Form", "Plain Steel"))
        self.comboBox_2.setItemText(6, _translate("Form", "Sand-coated BFRP"))
        self.comboBox_2.setItemText(7, _translate("Form", "Helically SC GFRP"))
        self.comboBox_2.setItemText(8, _translate("Form", "CFRP"))
        self.comboBox_2.setItemText(9, _translate("Form", "HFRP"))
        self.comboBox_2.setItemText(10, _translate("Form", "Epoxy Coated Steel"))
        self.comboBox_3.setItemText(0, _translate("Form", "Unconditional"))
        self.comboBox_3.setItemText(1, _translate("Form", "Wet and dry"))
        self.comboBox_3.setItemText(2, _translate("Form", "Freezing and thawing"))
        self.comboBox_3.setItemText(3, _translate("Form", "Alkaline solution"))
        self.comboBox_3.setItemText(4, _translate("Form", "Seawater"))
        self.comboBox_4.setItemText(0, _translate("Form", "Control"))
        self.comboBox_4.setItemText(1, _translate("Form", "Bond breaker"))
        self.comboBox_4.setItemText(2, _translate("Form", "Release agent"))
        self.comboBox_4.setItemText(3, _translate("Form", "Splatter"))
        self.label_10.setText(_translate("Form", "Failure mode:"))
        self.label_14.setText(_translate("Form", "Input features:"))

    def predict(self):
        # Load the data
        df = pd.read_excel(r"D:\Articles\Flexural pull_out\Machine - With Alireza - 2024.xlsx", sheet_name='tmax', header=0)
        y = df.iloc[:, 8].to_numpy().reshape((-1, 1))
        X = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].to_numpy()
        Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=0.7, random_state=42)

        # Train the models
        DT = DecisionTreeRegressor(random_state=0, max_depth=9, min_samples_leaf=2, min_samples_split=5, max_features='log2')
        RF = RandomForestRegressor(random_state=0, min_samples_leaf=1, min_samples_split=2, max_depth=None, max_features='sqrt', n_estimators=250)
        GB = GradientBoostingRegressor(learning_rate=0.15, n_estimators=200, max_depth=3, random_state=0)
        XGB = XGBRegressor(learning_rate=0.45, n_estimators=100, max_depth=3, random_state=0)

        # Fit the models
        DT.fit(Xtr, ytr)
        RF.fit(Xtr, ytr)
        GB.fit(Xtr, ytr)
        XGB.fit(Xtr, ytr)

        # Get the input values from the form
        concrete_type = self.comboBox.currentIndex()
        fc = float(self.lineEdit_3.text())
        bar_size = float(self.lineEdit_4.text())
        cmin = float(self.lineEdit.text())
        embedment_length = float(self.lineEdit_2.text())
        bar_type = self.comboBox_2.currentIndex()
        condition_type = self.comboBox_3.currentIndex()
        contaminant = self.comboBox_4.currentIndex()

        # Create the input feature array
        input_features = [concrete_type, fc, bar_size, cmin, embedment_length, bar_type, condition_type, contaminant]
        input_features = np.array(input_features).reshape(1, -1)

        # Select the model
        if self.radioButton.isChecked():
            model = DT
        elif self.radioButton_2.isChecked():
            model = RF
        elif self.radioButton_3.isChecked():
            model = GB
        elif self.radioButton_4.isChecked():
            model = XGB

        # Make predictions
        predicted_tmax = round(model.predict(input_features)[0], 2)

        # Determine failure mode
        if predicted_tmax < 1.0:
            predicted_failure_type = "Bond failure"
        elif 1.0 <= predicted_tmax < 2.0:
            predicted_failure_type = "Shear"
        else:
            predicted_failure_type = "Bar rupture"

        # Display the results in the line edits with larger and bold font
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)

        self.lineEdit_9.setFont(font)
        self.lineEdit_9.setText(str(predicted_tmax))

        self.lineEdit_10.setFont(font)
        self.lineEdit_10.setText(predicted_failure_type)


if __name__ == "__main__":
    import sys
    import numpy as np
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())


# In[ ]:




