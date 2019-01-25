# -*- coding:utf-8 -*-
import caffe
import cluster2d as ct
import numpy as np 
import featuregenerate as fg 
import validpoint as vp
import pointcloud as pc 
import matplotlib.pyplot as plt 
import matplotlib.patches as patch 

class CNNSegmention():
    def __init__(self):
        self.feature_blob=None
        self.outblobs=None
        self.vldpc=None
        self.cluster=None
    
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
        self.vldpc=vp.ValidPoints(pcdata=rawpoint,rows=640,cols=640,
                             lrange=60,max_height=5,min_height=-5)
        ffg=fg.FeatureGenerator()
        self.feature_blob=ffg.Generate(self.vldpc)
        # setp2 : load caffe model and forword
        cnnseg_net=caffe.Net(proto_path,caffe_model_path,caffe.TEST)
        cnnseg_net.blobs['data'].data[...]=self.feature_blob
        self.outblobs=cnnseg_net.forward()       


    def segment(self,object_thresh=0.5):
        clst=ct.Cluster2d(self.vldpc)
        clst.Cluster(self.outblobs,object_thresh)
        clst.Filter(self.outblobs)
        clst.Classify(self.outblobs)
        clst.GetObjects()
        self.cluster=clst
        return True

    def display_obstacle(self):
        obstacle_Map=np.zeros(self.vldpc.rows*self.vldpc.cols)
        max_min_rowcol=[]
        dic_obclass={'Unknow':100,'Vehicle':110,'Bicycle':120,'Pedestrian':130}
        for objc,idx in zip(self.cluster.objects,self.cluster.valid_objects):
            obstacle_Map[self.cluster.obstacles[idx].grids]=dic_obclass[objc.type]
            [r,c]=self.grid2rowcol(self.cluster.obstacles[idx].grids,self.vldpc.rows)
            r_max=np.max(r)
            r_min=np.min(r)
            c_max=np.max(c)
            c_min=np.min(c)
            max_min_rowcol.append([r_min,r_max,c_min,c_max])
        plt.imshow(obstacle_Map.reshape(self.vldpc.rows,self.vldpc.cols))
        ax=plt.gca()
        for idx in max_min_rowcol:
            rec=patch.Rectangle((idx[0],idx[2]),idx[3]-idx[2],idx[1]-idx[0],
                                edgecolor='r',facecolor='none')
            ax.add_patch(rec)
        plt.title('Obstacles Map')
        plt.show()

    @staticmethod
    def grid2rowcol(grid,rows):
        grid=np.array(grid)
        r=grid//rows-1
        c=grid%rows-1
        return r,c

if __name__=='__main__':
    proto_path='/home/reme/文档/cnnseg/Apollo_Lidar_CNNSeg/model/deploy.prototxt'
    caffe_model_path='/home/reme/文档/cnnseg/Apollo_Lidar_CNNSeg/model/deploy.caffemodel'
    test_pcd_path='/home/reme/文档/cnnseg/Apollo_Lidar_CNNSeg/data/test.pcd'
    test_cnnseg=CNNSegmention()
    test_cnnseg.forward(proto_path,caffe_model_path,test_pcd_path)
    ok=test_cnnseg.segment()
    if ok:
        print('Congratulations! \nAll the tasks have been completed and will generate obstacle information for you.')
        print('Valid objects index:',test_cnnseg.cluster.valid_objects)
        J=1
        for i in test_cnnseg.cluster.objects:
            print('%i' % J+':'+i.type,' score:%f'% i.score)
            J+=1
        print('Now, Will diplay obstacle map for you!')
        test_cnnseg.display_obstacle()
    else:
        print('Failed! \n Please Check it')
    