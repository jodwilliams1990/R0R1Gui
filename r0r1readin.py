"""
wget --user chec --password chec2048 https://www.mpi-hd.mpg.de/personalhomes/white/checs/data/d2018-05-14_DynamicRange_noNSB_5degC_gainmatched-200mV/Run43461_r0.tio
"""

#WITH WORKING R1 GUI

import sys
sys.path.append('../')
sys.path.append('../../')
from CHECLabPy.core.io import TIOReader
import os
import numpy as np
from matplotlib import pyplot as plt
from CHECLabPy.plotting.camera import CameraImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy, QWidget, QComboBox, QLabel, QRadioButton, \
    QCheckBox, QGridLayout, QLineEdit

input_dir= "/Users/chec/Desktop/CHECData/GainDropatdifferentPE"

def r0r1cameraplotter(input_path):
    reader = TIOReader(input_path)  # Load the file
    wfs = reader[1]
    for m in range (0,13):
        camera = plt.figure(figsize=(10, 10))
        # Generate a CameraImage object using the classmethod "from_mapping" which accepts the
        # mapping object contained in the reader, which converts from pixel ID to pixel
        # coordinates (using the Mapping class in TargetCalib)
        camera = CameraImage.from_mapping(reader.mapping)
        camera.add_colorbar()
        camera.image = wfs[:, m*10]  # Plot value of the sample at m x 10ns for each pixel
        plt.show()
        '''
        image=np.zeros((8,8))
        for k in range (0,7):
            for l in range (0,7):
                image[k,l]=allevents[(8*k)+l,10*m,0]
        fig = plt.figure(figsize=(10, 10))  # size of plot
        ax = fig.add_subplot(111)  # size of overall image
        im = ax.imshow(image, origin='lower')
        fig.colorbar(im)
        fig.show()
        '''

readalldatain=1
plotdata=0
plotfinal=0
plotr1oncam=0
avgplot=1
readinone=0
gui=2

if avgplot==1:
    # READ IN AND PLOT ALL DATA
    if readalldatain==1:
    # READ IN DATA FROM ALL EVENTS, ALL PIXELS, ALL TIME STAMPS
        allevents=np.zeros((64,128,11196))
        input_path = os.path.join(input_dir, "data_Run00559_r1.tio")  # The path to the r1 run file
        reader = TIOReader(input_path)  # Load the file
        for i in range(0,11196):
            ievent = i  # We will view the i-th event
            wfs = reader[ievent]  # Obtain the waveforms for the 10th event     #print("This event contains the samples for {} pixels and {} samples".format(wfs.shape[0], wfs.shape[1]))
            for j in range (0,64):
                allevents[j,:,i]=wfs[j]
    if plotdata==1:
    # PLOTTING AVERAGE ACROSS EVENTS AND TIME STAMPS
        a=sum(allevents)/64
        a=np.transpose(a)
        b=sum(a)/11196
        plt.plot(b,label='1000MHz,0.91PE, average over 11196 events, 64 pixels')
    if readalldatain==1:
    # READ IN DATA FROM ALL EVENTS, ALL PIXELS, ALL TIME STAMPS
        allevents=np.zeros((64,128,2218))
        input_path = os.path.join(input_dir, "data_Run00319_r1.tio")  #319 The path to the r1 run file
        reader = TIOReader(input_path)  # Load the file
        for i in range(0,2218):
            ievent = i  # We will view the 10th event
            wfs = reader[ievent]  # Obtain the waveforms for the 10th event     #print("This event contains the samples for {} pixels and {} samples".format(wfs.shape[0], wfs.shape[1]))
            for j in range (0,64):
                allevents[j,:,i]=wfs[j]
    if plotdata==1:
        # PLOTTING AVERAGE ACROSS EVENTS AND TIME STAMPS
        a=sum(allevents)/64
        a=np.transpose(a)
        b=sum(a)/2218
        plt.plot(b, label='0MHz,0.91PE, average over 2218 events, 64 pixels')
    if plotfinal==1:
        # PLOTTING AVERAGE ACROSS EVENTS AND TIME STAMPS
        plt.xlabel("Time (ns)")
        plt.ylabel("Signal (mV)")
        plt.legend(loc='best')
        plt.show()
if plotr1oncam==2:
    # PLOTTING DATA ON CAMERA FOR A SINGLE TIME STAMP FOR A GIVEN EVENT
    input_path = os.path.join(input_dir, "data_Run00505_r1.tio")  # 319 The path to the r1 run file
    r0r1cameraplotter(input_path)

if readinone==1:
    # READ IN A GIVEN EVENT FROM A GIVEN DATA RUN
    ievent=1
    input_path = os.path.join(input_dir, "data_Run00505_r1.tio")  # The path to the r1 run file
    reader = TIOReader(input_path)  # Load the file
    wfs = reader[ievent]  # Obtain the waveforms for the 10th event
    plt.plot(wfs[20], label='1000MHz,0PE,event 10')
    #(mu,sigma)=norm.fit(wfs)
    #snr[0]=mu/sigma
    #snr2[0]=sigma
    plt.legend(loc='upper left')
    plt.xlabel("Signal (mV)")
    plt.xlabel("Time (ns)")
    plt.show()

if gui==2:
    class App(QWidget):
        def __init__(self):
            super().__init__()
            self.left = 0
            self.top = 800
            self.title = 'R1 Live Data'
            self.width = 800
            self.height = 800
            self.initUI()
            self.plot()
            self.show()
        def initUI(self):
            self.setWindowTitle(self.title)
            self.setGeometry(self.left, self.top, self.width, self.height)
            l = QGridLayout(self)
            self.figure = plt.figure(figsize=(15, 5))
            self.canvas = FigureCanvas(self.figure)
            l.addWidget(self.canvas, 0, 0, 9, (100 - 4))
            self.compute_initial_figure()
            self.show()
        def compute_initial_figure(self):
            h=0
            '''
            xpix = reader.mapping['xpix'].values
            ypix = reader.mapping['ypix'].values
            size = reader.mapping.metadata['size']
            xpix = xpix / size + 3.5
            ypix = ypix / size + 3.5
            xpix = np.round(xpix)
            ypix = np.round(ypix)
            xpix2 = np.zeros(64, )
            '''
            image = np.zeros((8, 8))
            xpix = [5, 4, 5, 4, 6, 7, 7, 6, 4, 4, 5, 5, 6, 7, 7, 6, 6, 6, 7, 7, 4, 5, 4, 5, 6, 7, 7, 6, 4, 4, 5, 5, 3,
                    3, 2, 2, 1, 0, 0, 1, 2, 3, 2, 3, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 3, 2, 3, 2, 3, 3, 2, 2]
            ypix = [0, 0, 1, 1, 1, 1, 0, 0, 2, 3, 3, 2, 3, 3, 2, 2, 5, 4, 4, 5, 4, 5, 5, 4, 7, 7, 6, 6, 7, 6, 7, 6, 2,
                    3, 3, 2, 1, 0, 1, 0, 0, 1, 1, 0, 2, 3, 2, 3, 7, 7, 6, 6, 5, 4, 5, 4, 6, 6, 7, 7, 4, 5, 4, 5]
            image[xpix, ypix] = wfs[:, h]
            fig = plt.figure(figsize=(10, 10))  # size of plot
            ax = self.figure.add_subplot(111)  # size of overall image
            im = ax.imshow(image, origin='lower')
            fig.colorbar(im)
            im.set_clim(0, 10)
            self.canvas.draw()
            self.show()
        def plot(self):
            for v in range (1,128):
                QApplication.processEvents()
                print(v)
                time.sleep(0.1)
                plt.cla()


                '''
                xpix = reader.mapping['xpix'].values
                ypix = reader.mapping['ypix'].values
                size = reader.mapping.metadata['size']
                xpix = xpix / size + 3.5
                ypix = ypix / size + 3.5
                xpix = np.round(xpix)
                ypix = np.round(ypix)
                xpix2 = np.zeros(64, )
                '''
                image = np.zeros((8, 8))
                xpix = [5, 4, 5, 4, 6, 7, 7, 6, 4, 4, 5, 5, 6, 7, 7, 6, 6, 6, 7, 7, 4, 5, 4, 5, 6, 7, 7, 6, 4, 4, 5, 5, 3,
                        3, 2, 2, 1, 0, 0, 1, 2, 3, 2, 3, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 3, 2, 3, 2, 3, 3, 2, 2]
                ypix = [0, 0, 1, 1, 1, 1, 0, 0, 2, 3, 3, 2, 3, 3, 2, 2, 5, 4, 4, 5, 4, 5, 5, 4, 7, 7, 6, 6, 7, 6, 7, 6, 2,
                        3, 3, 2, 1, 0, 1, 0, 0, 1, 1, 0, 2, 3, 2, 3, 7, 7, 6, 6, 5, 4, 5, 4, 6, 6, 7, 7, 4, 5, 4, 5]
                image[xpix, ypix] = wfs[:, v]
                fig = plt.figure(figsize=(10, 10))  # size of plot
                ax = self.figure.add_subplot(111)  # size of overall image

                '''
                f1=ax.imshow(image) #,cmap=plt.cm.get_cmap('RdBu'))
                plt.colorbar(f1)
                f1.set_clim(0,500)
                '''
                im = ax.imshow(image, origin='lower')
                fig.colorbar(im)
                ax.set_title('%d ns' % v)
                im.set_clim(0,10)
                self.canvas.draw()
                self.show()
    if __name__ == '__main__':
        app = QApplication(sys.argv)
        w = App()
        app.exec_()