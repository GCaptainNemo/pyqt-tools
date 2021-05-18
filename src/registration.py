#coding=utf8
from PyQt5 import QtGui, QtWidgets, QtCore
from sys import argv, exit
import cv2
import threading
import os
import numpy as np
import matplotlib.pyplot as plt

class MyLoadingVideo(QtWidgets.QWidget):
    """
    Rewrite Qwidget to display movies.
    """
    def __init__(self):
        super(MyLoadingVideo, self).__init__()
        self.movie_dir = ""
        self.label = QtWidgets.QLabel('', self)
        self.label.setGeometry(QtCore.QRect(20, 40, 751, 401))
        self.videoflag = -1

    def setMovie(self):
        try:
            self.cap = cv2.VideoCapture(self.movie_dir)
            self.videoflag *= (-1)
            thu = threading.Thread(target=self.Display)
            thu.start()
            print("movie_dir = ", self.movie_dir)
        except Exception as e:
            print(e)

    def Display(self):
        while self.videoflag == 1:
            if self.cap.isOpened():
                success, frame = self.cap.read()
                self.picture = frame
                # RGB转BGR
                if success:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    img = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
                    self.label.setPixmap(QtGui.QPixmap.fromImage(img))
                    self.label.setScaledContents(True)
                    # time.sleep(0.1)
                    cv2.waitKey(10)
                else:
                    print("read failed, no frame data")
            else:
                print("open file or capturing device error, init again")
                self.reset()
            print(16)

    def stop(self):
        """ Slot function to stop the movie. """
        self.videoflag = -1


class MyWinPicture(QtWidgets.QWidget):
    def __init__(self):
        """
        Rewrite Qwidget to display pictures.
        param flag_:
        Two mode: 1. flag_ = 1, display through pictures address(self.dir)
                  2. flag_ = 0 display through pictures (self.picture_matrix)
        """
        super(MyWinPicture, self).__init__()
        self.pixmap = None

    def paintEvent(self, event):
        try:
            if self.pixmap is not None:
                painter = QtGui.QPainter(self)
                painter.drawPixmap(self.rect(), self.pixmap)
        except Exception as e:
            print(e)


class SegMent(QtCore.QThread):
    """ 为防止窗口3, 4点击'run'时程序卡死，使用多线程。 """
    def __init__(self, file_dir):
        super(SegMent, self).__init__()
        self.dir = file_dir
        self.run()

    def run(self):
        try:
            self.Segmentation()
        except Exception as e:
            print(e)

    def Segmentation(self):
        ...


class MyWindow(QtWidgets.QWidget):
    """
        __init__(): setup GUI and function connection
    """
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.show()
        self.resize(2000, 1500)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.setFont(font)
        self.setWindowTitle("Main Window")
        Tab = QtWidgets.QTabWidget()
        Tab.setFont(QtGui.QFont('Times New Roman', 10))
        self.window_1 = QtWidgets.QWidget()
        Tab.addTab(self.window_1, "数据集展示")
        # window2
        self.window_2 = QtWidgets.QWidget()
        Tab.addTab(self.window_2, "无人机数据评估")
        Vlayout = QtWidgets.QVBoxLayout(self.window_2)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.Movie_display_widget = MyLoadingVideo()
        self.textwidget_2_point = QtWidgets.QTextEdit()
        splitter.addWidget(self.Movie_display_widget)
        splitter.addWidget(self.textwidget_2_point)
        Vlayout.addWidget(splitter)
        Hlayout = QtWidgets.QHBoxLayout()
        self.button2_browse = QtWidgets.QPushButton("浏览")
        self.button2_browse.clicked.connect(self.browse_2)
        self.button2_stop = QtWidgets.QPushButton("停止")
        self.button2_stop.clicked.connect(self.stop_2)
        self.button2_point = QtWidgets.QPushButton("评分")
        self.button2_point.clicked.connect(self.point_2)
        Hlayout.addWidget(self.button2_browse)
        Hlayout.addWidget(self.button2_stop)
        Hlayout.addWidget(self.button2_point)
        Vlayout.addLayout(Hlayout)
        # Window3
        self.window_3 = QtWidgets.QWidget()
        Tab.addTab(self.window_3, "样例级/语义解析")
        Vlayout_3 = QtWidgets.QVBoxLayout(self.window_3)
        self.show_image_widget_3 = MyWinPicture()
        self.show_text_3 = QtWidgets.QTextEdit()
        self.show_text_3.setReadOnly(True)
        splitter_image_text_3 = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter_image_text_3.addWidget(self.show_image_widget_3)
        splitter_image_text_3.addWidget(self.show_text_3)
        Vlayout_3.addWidget(splitter_image_text_3)
        Hlayout_3_button = QtWidgets.QHBoxLayout()
        self.button3_1 = QtWidgets.QPushButton("选择")
        self.button3_1.clicked.connect(self.browse_3)
        self.button3_2 = QtWidgets.QPushButton("Run")
        self.button3_2.clicked.connect(self.run_3)
        self.button3_3 = QtWidgets.QPushButton("准确率")
        self.button3_3.clicked.connect(self.calculate_3)
        Hlayout_3_button.addWidget(self.button3_1)
        Hlayout_3_button.addWidget(self.button3_2)
        Hlayout_3_button.addWidget(self.button3_3)
        Vlayout_3.addLayout(Hlayout_3_button)
        # Window4
        self.window_4 = QtWidgets.QWidget()
        Tab.addTab(self.window_4, "样例级/语义解析")
        Vlayout_4 = QtWidgets.QVBoxLayout(self.window_4)
        self.show_image_widget_4_IR = MyWinPicture()
        self.show_image_widget_4_RGB = MyWinPicture()
        Vsplitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.show_text_4 = QtWidgets.QTextEdit()
        self.show_text_4.setReadOnly(True)
        splitter_image_text_4 = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        Vsplitter.addWidget(self.show_image_widget_4_IR)
        Vsplitter.addWidget(self.show_image_widget_4_RGB)

        splitter_image_text_4.addWidget(Vsplitter)
        splitter_image_text_4.addWidget(self.show_text_4)
        splitter_image_text_4.setSizes([100, 200])

        Vlayout_4.addWidget(splitter_image_text_4)
        Hlayout_4_button = QtWidgets.QHBoxLayout()
        self.button4_1 = QtWidgets.QPushButton("选择(IR)")
        self.button4_1.clicked.connect(self.browse_4_IR)
        self.button4_2 = QtWidgets.QPushButton("Run(IR)")
        self.button4_2.clicked.connect(self.run_4_IR)
        self.button4_3 = QtWidgets.QPushButton("准确率(IR)")
        self.button4_3.clicked.connect(self.calculate_4_IR)
        self.button4_4 = QtWidgets.QPushButton("选择(RGB)")
        self.button4_4.clicked.connect(self.browse_4_RGB)
        self.button4_5 = QtWidgets.QPushButton("Run(RGB)")
        self.button4_5.clicked.connect(self.run_4_RGB)
        self.button4_6 = QtWidgets.QPushButton("准确率(RGB)")
        self.button4_6.clicked.connect(self.calculate_4_RGB)

        Hlayout_4_button.addWidget(self.button4_1)
        Hlayout_4_button.addWidget(self.button4_2)
        Hlayout_4_button.addWidget(self.button4_3)
        Hlayout_4_button.addWidget(self.button4_4)
        Hlayout_4_button.addWidget(self.button4_5)
        Hlayout_4_button.addWidget(self.button4_6)

        Vlayout_4.addLayout(Hlayout_4_button)
        # Window5
        self.window_5 = QtWidgets.QWidget()
        VSplitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        Hsplitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        vlayout = QtWidgets.QVBoxLayout(self.window_5)
        self.Infrared_widget_5 = MyWinPicture()
        self.RGB_widget_5 = MyWinPicture()
        self.Output_5 = MyWinPicture()
        VSplitter.addWidget(self.Infrared_widget_5)
        VSplitter.addWidget(self.RGB_widget_5)
        Hsplitter.addWidget(VSplitter)
        Hsplitter.addWidget(self.Output_5)
        Hlayout = QtWidgets.QHBoxLayout()
        self.button5_1 = QtWidgets.QPushButton('Browse(IR)')
        self.button5_1.clicked.connect(self.browse5_IR)
        self.button5_2 = QtWidgets.QPushButton('Browse(RGB)')
        self.button5_2.clicked.connect(self.browse5_RGB)
        self.button5_3 = QtWidgets.QPushButton('Extract feature points')
        self.button5_3.clicked.connect(self.Extract_feature_points)

        self.button5_4 = QtWidgets.QPushButton('Registration')
        self.button5_4.clicked.connect(self.registration_5)

        Hlayout.addWidget(self.button5_1)
        Hlayout.addWidget(self.button5_2)
        Hlayout.addWidget(self.button5_3)
        Hlayout.addWidget(self.button5_4)
        Hsplitter.setSizes([100, 200])
        vlayout.addWidget(Hsplitter)
        vlayout.addLayout(Hlayout)
        Tab.addTab(self.window_5, "配准")
        # Window6
        self.window_6 = QtWidgets.QWidget()
        Tab.addTab(self.window_6, "三维模型展示")
        Vlayout = QtWidgets.QVBoxLayout(self.window_6)
        self.text_widget6 = QtWidgets.QTextEdit()
        Hlayout6_3 = QtWidgets.QHBoxLayout()
        self.button6_1 = QtWidgets.QPushButton("显示原始模型")
        self.button6_1.clicked.connect(self.show_orijinal_model6)
        self.button6_2 = QtWidgets.QPushButton("展示语义模型")
        self.button6_2.clicked.connect(self.show_semantic_model6)
        self.button6_3 = QtWidgets.QPushButton("展示实例模型")
        self.button6_3.clicked.connect(self.show_example_model6)
        self.button6_4 = QtWidgets.QPushButton("展示材质模型")
        self.button6_4.clicked.connect(self.show_material_model6)

        Hlayout6_3.addWidget(self.button6_1)
        Hlayout6_3.addWidget(self.button6_2)
        Hlayout6_3.addWidget(self.button6_3)
        Hlayout6_3.addWidget(self.button6_4)
        Vlayout.addWidget(self.text_widget6)
        Vlayout.addLayout(Hlayout6_3)
        # Window7
        self.window_7 = QtWidgets.QWidget()
        Tab.addTab(self.window_7, "仿真结果展示")
        Vlayout = QtWidgets.QVBoxLayout(self.window_7)
        self.text_widget7 = QtWidgets.QTextEdit()
        self.button7_openexe = QtWidgets.QPushButton("打开软件")
        self.button7_openexe.clicked.connect(self.Open_exe7)
        Vlayout.addWidget(self.text_widget7)
        Vlayout.addWidget(self.button7_openexe)
        Hlayout = QtWidgets.QHBoxLayout(self)
        Hlayout.addWidget(Tab)

    def browse_2(self):
        try:
            self.file_dir2, filetype = QtWidgets.QFileDialog.getOpenFileName(self, '选择文件', '',
                                                                                   'files(*.gif , *.avi, *.mp4)')
            print(self.file_dir2, filetype)
            self.Movie_display_widget.movie_dir = self.file_dir2
            self.Movie_display_widget.setMovie()
        except Exception as e:
            print(e)

    def stop_2(self):
        """ Stop the video. """
        try:
            self.Movie_display_widget.stop()
        except:
            pass

    def point_2(self):
        """ Show points. """
        try:
            ...
            # self.textwidget_2_point.setText("")
        except Exception as e:
            print(e)

    def browse_3(self):
        try:
            self.file_dir3, filetype = QtWidgets.QFileDialog.getOpenFileName(self, '选择文件', '',
                                                                                   'files(*.jpg , *.png)')
            print(self.file_dir3, filetype)
            pixmap = QtGui.QPixmap(self.file_dir3)
            self.show_image_widget_3.pixmap = pixmap
            self.show_image_widget_3.repaint()
        except Exception as e:
            print(e)

    def run_3(self):
        """ Window3 'Run' button. Use self.file_dir3 as an interface."""
        try:
            a = SegMent(self.file_dir3)
        except Exception as e:
            print(e)

    def calculate_3(self):
        """ Show results."""
        ...
        # self.show_text_3.setText(...)

    def browse_4_IR(self):
        """ Window4 browse Infra-Red picture"""
        try:
            self.file_dir4_IR, filetype = QtWidgets.QFileDialog.getOpenFileName(self, '选择文件', '',
                                                                            'files(*.jpg , *.png)')
            print(self.file_dir4_IR, filetype)
            pixmap = QtGui.QPixmap(self.file_dir4_IR)

            self.show_image_widget_4_IR.pixmap = pixmap
            self.show_image_widget_4_IR.repaint()
        except Exception as e:
            print(e)

    def browse_4_RGB(self):
        """ Window4 browse RGB picture"""
        try:
            self.file_dir4_RGB, filetype = QtWidgets.QFileDialog.getOpenFileName(self, '选择文件', '',
                                                                            'files(*.jpg , *.png)')
            print(self.file_dir4_RGB, filetype)
            pixmap = QtGui.QPixmap(self.file_dir4_RGB)

            self.show_image_widget_4_RGB.pixmap = pixmap
            self.show_image_widget_4_RGB.repaint()
        except Exception as e:
            print(e)

    def run_4_IR(self):
        """ Window4 'Run(IR)' button. Use self.file_dir4 as an interface."""
        try:
            a = SegMent(self.file_dir4_IR)
        except Exception as e:
            print(e)

    def run_4_RGB(self):
        """ Window4 'Run(RGB)' button. Use self.file_dir4 as an interface."""
        try:
            a = SegMent(self.file_dir4_RGB)
        except Exception as e:
            print(e)

    def calculate_4_IR(self):
        ...
        # self.show_text_3.setText(...)

    def calculate_4_RGB(self):
        ...
        # self.show_text_3.setText(...)

    def browse5_IR(self):
        try:
            self.file_dir5_IR, filetype = QtWidgets.QFileDialog.getOpenFileName(self, '选择红外图片', '',
                                                                            'files(*.jpg , *.png)')
            print(self.file_dir5_IR, filetype)
            self.Infrared_widget_5.pixmap = QtGui.QPixmap(self.file_dir5_IR)
            self.Infrared_widget_5.repaint()
        except Exception as e:
            print(e)

    def browse5_RGB(self):
        try:
            self.file_dir5_RGB, filetype = QtWidgets.QFileDialog.getOpenFileName(self, '选择可见光图片', '',
                                                                               'files(*.jpg , *.png)')
            print(self.file_dir5_IR, filetype)
            self.RGB_widget_5.pixmap = QtGui.QPixmap(self.file_dir5_RGB)
            self.RGB_widget_5.repaint()
        except Exception as e:
            print(e)

    def Extract_feature_points(self):
        try:
            self.feature_Point_lst_Ir = []
            self.feature_Point_lst_Rgb = []
            im_Ir = cv2.imread(self.file_dir5_IR)
            im_Bgr = cv2.imread(self.file_dir5_RGB)
            im_Rgb = cv2.cvtColor(im_Bgr, cv2.COLOR_BGR2RGB)
            fig = plt.figure('Manual registration')
            self.ax_Ir = plt.subplot(121), plt.imshow(im_Ir), plt.title('IR')
            self.ax_Rgb = plt.subplot(122), plt.imshow(im_Rgb), plt.title('RGB')
            self.ax_Ir[0].set_autoscale_on(False)
            self.ax_Rgb[0].set_autoscale_on(False)
            fig.canvas.mpl_connect("button_press_event", self.Press_Feature_Points)
            fig.canvas.mpl_connect("scroll_event", self.Scroll_zoom_in)
            fig.canvas.mpl_connect('key_press_event', self.on_key_press)
            plt.show()
        except Exception as e:
            print(e)

    def on_key_press(self, event):
        """ Use up, down, left, right to translate the picture."""
        try:
            axtemp = event.inaxes
            x_min, x_max = axtemp.get_xlim()
            y_min, y_max = axtemp.get_ylim()
            x_fanwei = (x_max - x_min) / 10
            y_fanwei = (y_max - y_min) / 10
            if event.key == 'up':
                axtemp.set(ylim=(y_min + y_fanwei, y_max + y_fanwei))
            elif event.key == 'down':
                axtemp.set(ylim=(y_min - y_fanwei, y_max - y_fanwei))
            elif event.key == 'left':
                axtemp.set(xlim=(x_min - x_fanwei, x_max - x_fanwei))
            elif event.key == 'right':
                axtemp.set(xlim=(x_min + x_fanwei, x_max + x_fanwei))
            elif event.key == 'ctrl+z':
                xlim = axtemp.get_xlim()
                ylim = axtemp.get_ylim()
                axtemp.cla()
                if axtemp == self.ax_Ir[0]:
                    self.feature_Point_lst_Ir.pop()
                    axtemp.imshow(cv2.imread(self.file_dir5_IR))
                    axtemp.set_title('IR')
                    for i, pos in enumerate(self.feature_Point_lst_Ir):
                        event.inaxes.scatter(pos[0], pos[1], c='orange', s=150, alpha=1.0, marker='*')
                        event.inaxes.text(pos[0], pos[1], i + 1, fontdict={'size': 20, 'color': 'red'})
                    axtemp.set(xlim=xlim, ylim=ylim)
                else:
                    self.feature_Point_lst_Rgb.pop()
                    axtemp.imshow(cv2.cvtColor(cv2.imread(self.file_dir5_RGB), cv2.COLOR_BGR2RGB))
                    axtemp.set_title('RGB')

                    for i, pos in enumerate(self.feature_Point_lst_Rgb):
                        event.inaxes.scatter(pos[0], pos[1], c='orange', s=150, alpha=1.0, marker='*')
                        event.inaxes.text(pos[0], pos[1], i + 1, fontdict={'size': 20, 'color': 'red'})
                    axtemp.set(xlim=xlim, ylim=ylim)
            axtemp.figure.canvas.draw_idle()
        except Exception as e:
            print(e)

    def Scroll_zoom_in(self, event):
        """ Scroll the mouse to zoom in(out) the picture."""
        try:
            axtemp = event.inaxes
            x_min, x_max = axtemp.get_xlim()
            y_min, y_max = axtemp.get_ylim()
            fanwei_x = (x_max - x_min) / 10
            fanwei_y = (y_max - y_min) / 10
            if event.button == 'up':
                axtemp.set(xlim=(x_min + fanwei_x, x_max - fanwei_x), ylim=(y_min + fanwei_y, y_max - fanwei_y))
            elif event.button == 'down':
                axtemp.set(xlim=(x_min - fanwei_x, x_max + fanwei_x), ylim=(y_min - fanwei_y, y_max + fanwei_y))
            event.inaxes.figure.canvas.draw_idle()  # 绘图动作实时反映在图像上
        except Exception as e:
            print(e)

    def Press_Feature_Points(self, event):
        """ Click the mouse to extract the featured points."""
        try:
            x = event.xdata
            y = event.ydata
            if event.inaxes == self.ax_Ir[0]:
                event.inaxes.scatter(x, y, c='orange', s=150, alpha=1.0, marker='*')
                self.feature_Point_lst_Ir.append([x, y])
                event.inaxes.text(x, y, str(len(self.feature_Point_lst_Ir)),
                                  fontdict={'size': 20, 'color': 'red'})
            else:
                event.inaxes.scatter(x, y, c='orange', s=150, alpha=1.0, marker='*')
                self.feature_Point_lst_Rgb.append([x, y])
                event.inaxes.text(x, y, str(len(self.feature_Point_lst_Rgb)),
                                        fontdict={'size': 20, 'color': 'red'})
            event.inaxes.figure.canvas.draw()
        except Exception as e:
            print(e)

    def registration_5(self):
        try:
            Im_bgr = cv2.imread(self.file_dir5_RGB)
            self.Im_gray = cv2.cvtColor(Im_bgr, cv2.COLOR_BGR2GRAY)
            self.Im_IR= cv2.imread(self.file_dir5_IR)[:, :, 1]
            Gray_pixel_pos = np.array(self.feature_Point_lst_Rgb, dtype='float32')
            IR_pixel_pos = np.array(self.feature_Point_lst_Ir, dtype='float32')
            Homography_matrix = cv2.findHomography(Gray_pixel_pos, IR_pixel_pos)
            dst = cv2.warpPerspective(self.Im_gray, Homography_matrix[0],
                                      tuple(list(self.Im_IR.shape)[::-1]))
            show_pair = np.stack([self.Im_IR, dst, self.Im_IR], axis=2)
            QtImg = QtGui.QImage(show_pair, show_pair.shape[1],
                                 show_pair.shape[0], QtGui.QImage.Format_RGB888)
            self.Output_5.pixmap = QtGui.QPixmap.fromImage(QtImg)
            self.Output_5.repaint()
        except Exception as e:
            print(e)

    def show_orijinal_model6(self):
        ...

    def show_semantic_model6(self):
        ...

    def show_example_model6(self):
        ...

    def show_material_model6(self):
        ...

    def Open_exe7(self):
        """ Function used to open .exe"""
        os.chdir("D:\\ENVI4.5\\") #子目录
        path_01 = "envi45winx86_32.exe"  #调用的exe
        os.system(path_01)
        print(1)


if __name__ == '__main__':
    try:
        app = QtWidgets.QApplication(argv)
        window = MyWindow()
        exit(app.exec_())
    except Exception as e:
        print(e)
