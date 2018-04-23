import json
import os
import numpy as np
import sys
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.Qt import QtCore, QtGui
import config_sim as conf

""" OBSOLETE """

class GLTextItem(GLGraphicsItem):
    def __init__(self, X=None, Y=None, Z=None, text=None, size=7):
        GLGraphicsItem.__init__(self)
        self.text = text
        self.X = X
        self.Y = Y
        self.Z = Z
        self.size = size

    def setGLViewWidget(self, GLViewWidget):
        self.GLViewWidget = GLViewWidget

    def setText(self, text):
        self.text = text
        self.update()

    def setX(self, X):
        self.X = X
        self.update()

    def setY(self, Y):
        self.Y = Y
        self.update()

    def setZ(self, Z):
        self.Z = ZrenderText
        self.update()

    def paint(self):
        self.GLViewWidget.qglColor(QtCore.Qt.white)
        self.myFont = QtGui.QFont("Helvetica", self.size);

        self.GLViewWidget.renderText(self.X, self.Y, self.Z, self.text,self.myFont)


class DET_SHOW(object):
    def __init__(self,data):
        self.sipm       = data['SIPM']['size']
        self.app = pg.QtGui.QApplication([])
        self.w   = gl.GLViewWidget()


    def np2cart(self,p):
        return {'x':p[0],'y':p[1],'z':p[2]}
    def cart2np(self,p):
        return [p['x'],p['y'],p['z']]
    def cyl2np(self,p):
        return [p['r'],p['z'],p['phi']]
    def np2cyl(self,p):
        return {'r':p[0],'z':p[1],'phi':p[2]}

    def cart2cyl(self,p):
        #x=p[0];y=p[1];z=p[2]
        q={}
        q['r'] = np.sqrt(np.square(p['x'])+np.square(p['y']))
        q['z'] = p['z']
        q['phi'] = np.arctan2(p['y'],p['x'])
        return q

    def cyl2cart(self,p):
        q={}
        q['z'] = p['z']
        q['x'] = p['r']*np.cos(p['phi'])
        q['y'] = p['r']*np.sin(p['phi'])
        return q

    def rot_cyl(self,p_array,phi):
        r = np.zeros(p_array.shape)
        for i in range(p_array.shape[0]):
            r[i,:] = self.cyl2np(self.cart2cyl(self.np2cart(p_array[i,:])))
            r[i,2] += phi
            r[i,:] = self.cart2np(self.cyl2cart(self.np2cyl(r[i,:])))
        return r


    def SiPM_QT(self,position,phi,name,photons,max_photons,id=False,
                show_photons=True):
        s = np.array(self.sipm,dtype=float)
        p = np.array([[s[0]/2,s[1]/2,s[2]/2],
                      [s[0]/2,s[1]/2,-s[2]/2],
                      [-s[0]/2,s[1]/2,s[2]/2],
                      [-s[0]/2,s[1]/2,-s[2]/2],
                      [s[0]/2,-s[1]/2,s[2]/2],
                      [s[0]/2,-s[1]/2,-s[2]/2],
                      [-s[0]/2,-s[1]/2,s[2]/2],
                      [-s[0]/2,-s[1]/2,-s[2]/2]])

        p=self.rot_cyl(p,phi)
        position3d = np.array([position for i in range(p.shape[0])])
        p = p + position3d
        verts = [ [p[0],p[1],p[3],p[2]],
              [p[0],p[2],p[6],p[4]],
              [p[1],p[3],p[7],p[5]],
              [p[1],p[0],p[4],p[5]],
              [p[6],p[4],p[5],p[7]],
              [p[6],p[2],p[3],p[7]]
            ]

        #plt = gl.GLScatterPlotItem(pos=p, color=pg.glColor('w'),size=1)
        meshdata = gl.MeshData(vertexes=p,faces=np.array([  [0,1,4],
                                                            [1,4,5],
                                                            [0,4,6],
                                                            [0,6,2],
                                                            [0,2,3],
                                                            [0,1,3],
                                                            [6,2,3],
                                                            [6,3,7],
                                                            [7,3,5],
                                                            [1,3,5]
                                                         ]))

        color = int((photons/max_photons)*200.0)
        color = pg.glColor(color+55,color+55,color+55)

        plt = gl.GLMeshItem(meshdata=meshdata, color = color,
                            #edgeColor=pg.glColor('r'),
                            drawEdges=False,
                            smooth=True,
                            shader='shaded',
                            glOptions='opaque',
                            computeNormals='False')
        self.w.addItem(plt)

        if (id==True) :
            t = GLTextItem( X=position[0],
                            Y=position[1],
                            Z=position[2],
                            text=str(int(name)))

            t.setGLViewWidget(self.w)
            self.w.addItem(t)

        if (show_photons==True and photons>0) :
            t = GLTextItem( X=position[0],
                            Y=position[1],
                            Z=position[2],
                            text=str(int(photons)))

            t.setGLViewWidget(self.w)
            self.w.addItem(t)


    def __call__(self,sensors,data,event,ident=False,show_photons=True):
        items = []
        for i in list(self.w.items):
            self.w.removeItem(i)

        #os.chdir(path)
        #self.sensors = pd.read_hdf(file_name,key='sensors')
        #print self.sensors
        #self.positions = np.array(self.sensors)
        #data = pd.read_hdf(file_name,key='MC')
        #data = np.array(data, dtype = 'int32')

        max_light = float(data[event,:].max())

        count=0
        for i in sensors:
            color = int((data[event,count]/max_light)*200.0)
            #print color
            self.SiPM_QT(i[1:].transpose(),
                         np.arctan2(i[2],i[1]),i[0],
                         data[event,count],
                         max_light,
                         id = ident,
                         show_photons=show_photons
                         )
            count+=1

        t = GLTextItem( X=0, Y=0, Z=0, text=str(event), size=12)
        t.setGLViewWidget(self.w)
        self.w.addItem(t)

        self.w.opts['distance']=500
        self.w.show()

        pg.QtGui.QApplication.exec_()



if __name__ == '__main__':

    # SIM=conf.SIM_DATA(read=False)
    # print SIM.data
    # A = DET_SHOW(SIM.data)
    #
    # os.chdir("/home/viherbos/DAQ_DATA/NEUTRINOS/RING/")
    # filename = "p_FRSET_0.h5"
    #
    # positions = np.array(pd.read_hdf(filename,key='sensors'))
    # data = np.array(pd.read_hdf(filename,key='MC'), dtype = 'int32')
    #
    # for i in range(0,100):
    #     A(positions,data,i,True,False)
    #A(positions,data,0,True,False)



    SIM_CONT=conf.SIM_DATA(filename="sim_config.json",read=True)
    print SIM_CONT.data
    B = DET_SHOW(SIM_CONT.data)

    os.chdir("/home/viherbos/DAQ_DATA/NEUTRINOS/Small_Animal/")
    filename = "p_FR_infinity_0.h5"

    positions = np.array(pd.read_hdf(filename,key='sensors'))
    data = np.array(pd.read_hdf(filename,key='MC'), dtype = 'int32')
    # for i in range(0,100):
    #     B(positions,data,i)
    B(positions,data,1,True,False)
