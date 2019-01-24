# -*- coding:utf-8 -*-
import caffe
import cluster2d as ct
import numpy as np 
import featuregenerate as fg 
import validpoint as vp
import pointcloud as pc 

class CNNSegmention():
    def __init__(self,rows=640,cols=640,lrange=60,
                 max_height=5,min_height=-5):
        self.feature_blob=None
        self.outblobs=None
        self.objects=None
        self.rows=rows
        self.cols=cols
        self.lrange=lrange
        self.max_height=max_height
        self.min_height=min_height
        self.vldpc=None
    
    def forward(self,proto_path,caffe_model_path,
                PCD_Path,USE_CAFFE_GPU=False):
        if USE_CAFFE_GPU:
            caffe.set_mode_gpu()
            caffe.set_device(0)
        else:
            caffe.set_mode_cpu()
        # step1 : feature_generate
        rawpoint=pc.PointCloud()
        rawpoint.ReadFromBinFile(PCD_Path)
        vldpc=vp.ValidPoints(rawpoint,self.rows,self.cols,
                            self.max_height,self.min_height,self.lrange)
        ffg=fg.FeatureGenerator(vldpc)
        self.feature_blob=ffg.Generate()
        self.vldpc=vldpc
        # setp2 : load caffe model and forword
        cnnseg_net=caffe.Net(proto_path,caffe_model_path,caffe.TEST)
        cnnseg_net.blobs['data'].data[...]=self.feature_blob
        self.outblobs=cnnseg_net.forward()       

    def segment(self,object_thresh=0.5):
        clst=ct.Cluster2d(self.vldpc,self.rows,self.cols,self.lrange)
        clst.Cluster(self.outblobs,object_thresh)
        clst.Filter(self.outblobs)
        clst.Classify(self.outblobs)
        clst.GetObjects()
        self.objects=clst.objects
        return True

       
if __name__=='__main__':
    proto_path='/home/reme/文档/cnnseg/Apollo_Lidar_CNNSeg/model/deploy.prototxt'
    caffe_model_path='/home/reme/文档/cnnseg/Apollo_Lidar_CNNSeg/model/deploy.caffemodel'
    test_pcd_path='/home/reme/文档/cnnseg/Apollo_Lidar_CNNSeg/data/test.pcd'
    test_cnnseg=CNNSegmention()
    test_cnnseg.forward(proto_path,caffe_model_path,test_pcd_path)
    test_cnnseg.segment()
    pass