# -*- coding: utf-8 -*-
'''
1.导入模块，简称pc
2.类PointCloud()
属性：
            x : x坐标
            y : y坐标
            z : z方向高度
    intensity : 反射强度
方法：
    xyz=PointCloud.ReadFromPcdFile(pcd_file_path)
    读取pcd二进制文件，返回numpy.array()类数据，shape=(len,4)
3.验证是否通过调试 ：True

'''

import pcl.io as pio 
import numpy as np 

class PointCloud():
    def __init__(self):
        self.x=None
        self.y=None
        self.z=None
        self.intensity=None

    def ReadFromPcdFile(self,pcd_file_path):
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

    def ReadFromBinFile(self,bin_file_path):
        pc=np.fromfile(bin_file_path,dtype=np.float32)
        pc=pc.reshape((pc.size//4,4))
        self.x=pc[:,0]
        self.y=pc[:,1]
        self.z=pc[:,2]
        self.intensity=pc[:,3]
        return pc


if __name__=='__main__':
    test_pcd=PointCloud()
    a=test_pcd.ReadFromPcdFile('/home/reme/桌面/CNNSeg/data/test.pcd')
    pass


