from PyQt5 import QtGui, QtWidgets, QtCore
from sys import argv, exit
import cv2
import threading
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# import pydensecrf.densecrf as dcrf
# from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import cv2
# from PIL import Image
# from utils.metrics import compute_iou_batch
# from models.net import EncoderDecoderNet, SPPNet
# from dataset.cityscapes import CityscapesDataset
# from dataset.mydata_temp import MyDataset
# from utils.preprocess import minmax_normalize
# import yaml


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
    """
    Rewrite Qwidget to display pictures.
    """
    def __init__(self):
        super(MyWinPicture, self).__init__()
        self.dir = None

    def paintEvent(self, event):
        try:
            painter = QtGui.QPainter(self)
            pixmap = QtGui.QPixmap(self.dir)
            painter.drawPixmap(self.rect(), pixmap)
        except Exception as e:
            print(e)

class MyWidgetRegistration(MyWinPicture):
    """
        Rewrite MyWinPicture to registrate manually.
    """
    def __init__(self):
        super(MyWidgetRegistration, self).__init__()
        self.Feature_Point_lst = []
        self.flag = 0

    def paintEvent(self, event):
        super().paintEvent(event)
        rect = QtCore.QRect(self.x0, self.y0, abs(self.x1 - self.x0), abs(self.y1 - self.y0))
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QPen(QtCore.Qt.red, 2, QtCore.Qt.SolidLine))
        painter.drawRect(rect)

    def mousePressEvent(self, event):
        self.x0 = event.x()
        self.y0 = event.y()
        self.Feature_Point_lst.append([event.x(), event.y()])
        self.flag = 1

    def mouseMoveEvent(self, event):
        if self.flag:
            self.x1 = event.x()
            self.y1 = event.y()
            self.update()


    def mouseReleaseEvent(self, event, points):
        try:
            # for p in self.lastClicked:
            #     p.resetPen()
            # for p in points:
            #     p.setPen('r', width=2)
            #     # self.signal_judge_point1.emit(mousePoint1.x(), mousePoint1.y())
            # self.lastClicked = points
            self.flag = 0
        except Exception as e:
            print(e)




class SegMent(QtCore.QThread):
    """ 为防止窗口3, 4点击'run'时程序卡死，使用多线程。 """
    def __init__(self, file_dir,a):
        super(SegMent, self).__init__()
        self.dir = file_dir#batch文件
        self.result=file_dir[1]#输出
        self.acc=0;#mean_accuracy
        self.flag = a
        self.run()

    def run(self):
        try:
            if self.flag == 1:
                self.Segmentation_rgb()
            else:
                self.Segmentation_ir()
        except Exception as e:
            print(e)

    def Segmentation_rgb(self):
        self.palette = [128, 64, 128, 244, 35, 232, 220, 20, 60, 255, 0, 0, 0, 0, 70, 0, 0, 142, 70, 70, 70, 102, 102, 156,
                   107, 142, 35, 152, 251, 152, 220, 220, 0, 70, 130, 180]
        zero_pad = 256 * 3 - len(self.palette)
        for i in range(zero_pad):
            self.palette.append(0)
        self.colorseg()
        print(1)
    def Swgmentation_ir(self):
        self.irseg()

    def colorize_mask(self,mask):
        # mask: numpy array of the mask
        print(mask.shape)
        new_mask = Image.fromarray(mask.astype(np.uint8).squeeze(), mode='L').convert('P')
        new_mask.putpalette(self.palette)
        return new_mask

    def dense_crf(img, output, classes):
        h = img.shape[0]
        w = img.shape[1]

        # output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        temp = np.zeros((classes, h, w))
        for c in range(classes):
            mask = output == c
            temp[c, :, :] = mask.astype(np.uint8)
        output = temp
        rROI = output == 0
        output[rROI] = 0.01
        U = -np.log(output)
        U = U.reshape((classes, -1))
        U = np.ascontiguousarray(U)
        U = U.astype(np.float32)
        # U = unary_from_softmax(output)
        d = dcrf.DenseCRF2D(w, h, classes)
        # print(np.min(U),np.max(U))

        img = np.ascontiguousarray(img)
        d.setUnaryEnergy(U)

        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=20, srgb=3, rgbim=img, compat=10)  # pairwise energy
        # pairwise_energy = create_pairwise_bilateral(sdims=(10,10), schan=(0.01,), img=img, chdim=2)
        # d.addPairwiseEnergy(pairwise_energy, compat=10)

        Q = d.inference(10)
        Q = np.argmax(np.array(Q), axis=0).reshape((h, w))
        # print(Q.max())
        return Q


    def colorseg(self):
        output_dir = '../model' + '/rgb_temp_crop'
        save_dir = output_dir
        net_config = {}
        data_config = {}
        batch_size = 1
        classes = np.arange(0, 7)

        net_config['output_channels'] = 7
        net_config['enc_type'] = 'xception65'
        net_config['dec_type'] = 'aspp'

        data_config['target_size'] = (432, 768)
        target_size = (2160, 3840)  # (684, 1216)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if 'unet' in net_config['dec_type']:
            net_type = 'unet'
            model = EncoderDecoderNet(**net_config).to(device)
        else:
            net_type = 'deeplab'
            model = SPPNet(**net_config).to(device)

        model = nn.DataParallel(model)
        model_path = save_dir + '/model.pth'
        print(model_path)
        param = torch.load(model_path, map_location=device)
        model.load_state_dict(param)
        del param

        Dataset = MyDataset
        # print('len(test_dataset):', len(test_dataset))
        # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        valid_loaders = []
        for iii in range(2160 // target_size[0]):
            for jjj in range(3840 // target_size[1]):
                valid_dataset = Dataset(split='test', net_type='deeplab', target_size=target_size, crop_pos=(iii, jjj))
                valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, pin_memory=True)
                valid_loaders.append(valid_loader)
        test_loader = valid_loaders[0]
        accu_list = []
        accu_list0 = []
        iou_list = []
        images_list = []
        labels_list = []
        preds_list = []

        model.eval()
        count = 0
        with torch.no_grad():
                count += 1
                path = 'result/{}.bmp'.format(count)
                images, labels, img_o = self.dir    #***********************************单纯在此处替换一下
                images_np = img_o.squeeze().numpy()
                labels_np = labels.squeeze().numpy()

                images, labels = images.to(device), labels.to(device)
                # preds = model.tta(images, net_type='deeplab')
                preds = F.interpolate(model(images), size=labels.shape[1:], mode='bilinear', align_corners=True)
                preds_soft_np = preds.detach().squeeze().cpu().numpy()
                preds = preds.argmax(dim=1)
                preds_np = preds.detach().squeeze().cpu().numpy()
                preds_np = self.dense_crf(images_np.astype(np.uint8), preds_np, 7)
                # preds_np = dense_crf(images_np.astype(np.uint8),preds_soft_np,7)

                ROI = (labels_np != 255)
                mask = preds_np == 1
                preds_np[mask] = 5
                mask = labels_np == 1
                labels_np[mask] = 5
                mask = preds_np == 0
                preds_np[mask] = 5
                mask = labels_np == 0
                labels_np[mask] = 5
                temp = preds_np[ROI] == labels_np[ROI]

                accu = np.sum(temp) / (np.sum(ROI) + (np.sum(ROI) == 0))
                iou = compute_iou_batch(preds_np, labels_np, classes)  # 计算交并比
                if np.isnan(iou):
                    print("continue")
                print(count, np.sum(temp), np.sum(ROI), accu, iou)
                accu_list.append(accu)
                iou_list.append(iou)

                images_list.append(np.sum(temp))
                labels_list.append(np.sum(ROI))
                preds_list.append(preds_np)

                if accu > 0.00:
                    accu_list0.append(accu)
                    ignore_pixel = labels_np == 255
                    # preds_np[ignore_pixel] = 7
                    labels_np[ignore_pixel] = 7
                    preds_new = self.colorize_mask(preds_np)
                    labels_new = labels_np
                    self.result=preds_new
                    # print(preds_new.shape)
                    # Image.fromarray((images_np).astype(np.uint8)).save(path)
                    # Image.fromarray((preds_new*255/7).astype(np.uint8)).save(path.replace('.bmp', 'pred.bmp'))
                    # Image.fromarray((labels_new*255/7).astype(np.uint8)).save(path.replace('.bmp', 'label.bmp'))
                    # cv2.imwrite(path.replace('.bmp', 'pred.bmp'), preds_new*255/7)
                    # cv2.imwrite(path.replace('.bmp', 'label.bmp'), labels_new*255/7)

        mean_accu0 = np.mean(accu_list0)
        # mean_iou = np.mean(iou_list)
        mean_accu = np.sum(images_list) / np.sum(labels_list)
        self.acc=mean_accu

    def irseg(self):
        output_dir = '../model'+ '/IR_temp'
        save_dir = output_dir
        net_config = {}
        data_config = {}
        batch_size = 1
        classes = np.arange(0, 6)

        net_config['output_channels'] = 6
        net_config['enc_type'] = 'xception65'
        net_config['dec_type'] = 'aspp'

        data_config['target_size'] = (432,768)
        target_size = (480,720)#(684, 1216)

        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


        if 'unet' in net_config['dec_type']:
            net_type = 'unet'
            model = EncoderDecoderNet(**net_config).to(device)
        else:
            net_type = 'deeplab'
            model = SPPNet(**net_config).to(device)

        #model = nn.DataParallel(model)
        model_path = save_dir + '/model.pth'
        print(model_path)
        param = torch.load(model_path,map_location=device)
        model.load_state_dict(param)
        del param


        Dataset = MyDataset
        # print('len(test_dataset):', len(test_dataset))
        # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        valid_loaders = []
        valid_dataset = Dataset(split='test', net_type='deeplab', target_size=target_size)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        valid_loaders.append(valid_loader)

        accu_list = []
        accu_list0 = []
        iou_list = []
        images_list = []
        labels_list = []
        preds_list = []

        model.eval()
        with torch.no_grad():
            for test_loader in valid_loaders:
                count = 0
                for batched in test_loader:
                    count += 1
                    path = 'result/IR{}.bmp'.format(count)
                    images, labels, img_o = self.dir
                    images_np = img_o.squeeze().numpy()
                    labels_np = labels.squeeze().numpy()

                    images, labels = images.to(device), labels.to(device)
                    # preds = model.tta(images, net_type='deeplab')
                    preds = F.interpolate(model(images), size=labels.shape[1:], mode='bilinear', align_corners=True)
                    preds_soft_np = preds.detach().squeeze().cpu().numpy()
                    preds = preds.argmax(dim=1)
                    preds_np = preds.detach().squeeze().cpu().numpy()
                    preds_np = self.dense_crf(images_np.astype(np.uint8),preds_np,6)
                    # preds_np = dense_crf(images_np.astype(np.uint8),preds_soft_np,7)

                    ROI = (labels_np != 255)
                    temp = preds_np[ROI] == labels_np[ROI]

                    accu = np.sum(temp)/(np.sum(ROI)+(np.sum(ROI)==0))
                    iou = compute_iou_batch(preds_np, labels_np, classes)    # 计算交并比
                    if np.isnan(iou):
                        continue
                    print(count,np.sum(temp), np.sum(ROI), accu, iou)
                    accu_list.append(accu)
                    iou_list.append(iou)

                    images_list.append(np.sum(temp))
                    labels_list.append(np.sum(ROI))
                    preds_list.append(preds_np)

                    if accu>0.00:
                        accu_list0.append(accu)
                        ignore_pixel = labels_np == 255
                        preds_np[ignore_pixel] = 6
                        labels_np[ignore_pixel] = 6
                        preds_new = preds_np
                        labels_new = labels_np
                        self.result=preds_np
        mean_accu0 = np.mean(accu_list0)
        #mean_iou = np.mean(iou_list)
        mean_accu = np.sum(images_list)/np.sum(labels_list)
        self.acc = mean_accu


class MyWindow(QtWidgets.QWidget):
    """
        __init__(): setup GUI and function connection
    """
    def __init__(self):
        super(MyWindow, self).__init__()
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

        self.accir = 0; self.accrgb = 0

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
        # self.Infrared_widget_5 = MyWinPicture()
        # self.RGB_widget_5 = MyWinPicture()
        # self.Output_5 = MyWinPicture()
        self.Infrared_widget_5 = MyWidgetRegistration()
        self.RGB_widget_5 = MyWidgetRegistration()
        self.Output_5 = MyWidgetRegistration()
        VSplitter.addWidget(self.Infrared_widget_5)
        VSplitter.addWidget(self.RGB_widget_5)
        Hsplitter.addWidget(VSplitter)
        Hsplitter.addWidget(self.Output_5)
        Hlayout = QtWidgets.QHBoxLayout()
        self.button5_1 = QtWidgets.QPushButton('Browse(IR)')
        self.button5_1.clicked.connect(self.browse5_IR)
        self.button5_2 = QtWidgets.QPushButton('Browse(RGB)')
        self.button5_2.clicked.connect(self.browse5_RGB)
        self.button5_3 = QtWidgets.QPushButton('Registration')
        self.button5_3.clicked.connect(self.registration_5)

        Hlayout.addWidget(self.button5_1)
        Hlayout.addWidget(self.button5_2)
        Hlayout.addWidget(self.button5_3)
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
            self.show_image_widget_3.dir = self.file_dir3
            self.show_image_widget_3.update()
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
            self.show_image_widget_4_IR.dir = self.file_dir4_IR
            self.show_image_widget_4_IR.update()
        except Exception as e:
            print(e)

    def browse_4_RGB(self):
        """ Window4 browse RGB picture"""
        try:
            self.file_dir4_RGB, filetype = QtWidgets.QFileDialog.getOpenFileName(self, '选择文件', '',
                                                                            'files(*.jpg , *.png)')
            print(self.file_dir4_RGB, filetype)
            self.show_image_widget_4_RGB.dir = self.file_dir4_RGB
            self.show_image_widget_4_RGB.update()
        except Exception as e:
            print(e)

    def run_4_IR(self):
        """ Window4 'Run(IR)' button. Use self.file_dir4 as an interface."""
        try:
            a = SegMent(self.file_dir4_IR,0)
            self.show_image_widget_4_IR.dir = a.result
            self.accir = a.acc
            self.show_image_widget_4_IR.update()
        except Exception as e:
            print(e)

    def run_4_RGB(self):
        """ Window4 'Run(RGB)' button. Use self.file_dir4 as an interface."""
        try:
            a = SegMent(self.file_dir4_RGB,1)
            self.show_image_widget_4_RGB.dir = a.result
            self.accrgb=a.acc
            self.show_image_widget_4_RGB.update()

        except Exception as e:
            print(e)

    def calculate_4_IR(self):
        self.show_text_3.setText(self.accir)

    def calculate_4_RGB(self):
        self.show_text_3.setText(self.accrgb)

    def browse5_IR(self):
        try:
            self.file_dir5_IR, filetype = QtWidgets.QFileDialog.getOpenFileName(self, '选择红外图片', '',
                                                                            'files(*.jpg , *.png)')
            print(self.file_dir5_IR, filetype)
            self.Infrared_widget_5.dir = self.file_dir5_IR
            self.Infrared_widget_5.update()
        except Exception as e:
            print(e)

    def browse5_RGB(self):
        try:
            self.file_dir5_RGB, filetype = QtWidgets.QFileDialog.getOpenFileName(self, '选择可见光图片', '',
                                                                            'files(*.jpg , *.png)')
            print(self.file_dir5_RGB, filetype)
            self.RGB_widget_5.dir = self.file_dir5_RGB
            self.RGB_widget_5.update()
        except Exception as e:
            print(e)

    def registration_5(self):
        ...

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
