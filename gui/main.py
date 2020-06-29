'''
@Author       : Scallions
@Date         : 2020-06-05 12:55:36
@LastEditors  : Scallions
@LastEditTime : 2020-06-09 23:31:13
@FilePath     : /gps-ts/gui/main.py
@Description  : 
'''

# from PyQt5 import QApplication, QMainWindow

import sys
from PySide2.QtWidgets import QApplication, QLabel


app = QApplication(sys.argv)
label = QLabel("<font color=red size=40>Hello World!</font>")
label.show()
app.exec_()



