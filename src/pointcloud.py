# -*- coding: utf-8 -*-
import pcl.io as pio 
import numpy as np 

class PointCloud():
    def __init__(self):
        self.x=None
        self.y=None
        self.z=None
        self.intensity=None

    def ReadFromBinFile(self,pcd_file_path):
        reader=pio.PCDReader()
        pcdata=reader.read(pcd_file_path)
        self.x=pcdata[0].data['x']
        self.y=pcdata[0].data['y']
        self.z=pcdata[0].data['z']
        self.intensity=pcdata[0].data['intensity']
        return np.concatenate([self.x.reshape(self.x.size,1),
                               self.y.reshape(self.y.size,1),
                               self.z.reshape(self.z.size,1),
                               self.intensity.reshape(self.intensity.size,1)],axis=1)

if __name__=='__main__':
    test_pcd=PointCloud()
    a=test_pcd.ReadFromBinFile('/home/reme/桌面/CNNSeg/data/test.pcd')
    pass


