# -*- coding:utf-8 -*-
import numpy as np 
import node as nd 
import util as ut 
import obstacle as obs 
import objects as obj

class Cluster2d():
    def __init__(self,vldpc,rows,cols,lrange):
        self.obstacles=[]
        self.objects=[]
        self.rows=rows
        self.cols=cols
        self.node=None
        self.lrange=lrange
        self.count_obstacles=0
        self.id_img=-np.ones(self.rows*self.cols)
        self.vldpc=vldpc

    def Cluster(self,outblobs,object_thresh=0.5):
        category_pt_data=outblobs['category_score'][0,0] #ndim=2
        instance_pt_x=outblobs['instance_pt'][0,0]
        instance_pt_y=outblobs['instance_pt'][0,1]
        self.node=nd.Node(self.rows,self.cols)
        self.node.point_num=self.GetPointsNum()
        self.node.is_object=np.all([self.node.point_num>0,
                                    category_pt_data>=object_thresh],axis=0)
        gird_col,grid_row=np.meshgrid(np.arange(self.cols),np.arange(self.rows))
        scale=0.5*self.rows/self.lrange
        center_row=np.round(grid_row+instance_pt_x*scale).astype(np.int)
        center_col=np.round(gird_col+instance_pt_y*scale).astype(np.int)
        center_row=np.where(center_row>=0,center_row,0)
        center_row=np.where(center_row<self.rows,center_row,self.rows-1)
        center_col=np.where(center_col>=0,center_col,0)
        center_col=np.where(center_col<self.cols,center_col,self.cols-1)
        self.node.center_node[0]=center_row
        self.node.center_node[1]=center_col

        for row in range(self.rows):
            for col in range(self.cols):
                if self.node.is_object[row,col] and \
                   self.node.traversed[row,col]==0:
                    ut.Traverse(self.node,row,col)

        for row in range(self.rows):
            for col in range(self.cols):
                if (not self.node.is_center[row,col]):
                    continue
                for row2 in [row-1,row,row+1]:
                    for col2 in [col-1,col,col+1]:
                        if (row2==row or col2==col) and \
                            (row2>=0 and row2<self.rows and col2>=0 
                             and col2<self.cols):
                            if self.node.is_center[row2,col2]:
                                ut.SetUnion(self.node,row,col,row2,col2)

        for row in range(self.rows):
            for col in range(self.cols):
                if (not self.node.is_object[row,col]):
                    continue
                r_row,r_col=ut.SetFind(self.node,row,col)
                if self.node.obstacle_id[r_row,r_col]<0:
                    self.node.obstacle_id[r_row,r_col]=self.count_obstacles
                    self.count_obstacles+=1
                    self.obstacles.append(obs.Obstacle())
                grid=row*self.rows+col
                self.id_img[grid]=self.node.obstacle_id[r_row,r_col]
                self.obstacles[self.node.obstacle_id[r_row,r_col]].grids.append(grid)

    def GetPointsNum(self):
        idx_row=self.vldpc.idx_row
        idx_col=self.vldpc.inidx_col
        map_idx=idx_row*self.cols+idx_col
        unidx,counts=np.unique(map_idx,return_counts=True)
        grid_count=np.zeros(self.rows*self.cols)
        for idx,count in zip(unidx,counts):
            grid_count[idx]=count
        return grid_count.reshape((self.rows,self.cols))

    def Filter(self,outblobs):
        confidence_pt_data=outblobs['confidence_score'].reshape((self.rows*self.cols,))
        height_pt_data=outblobs['height_pt'].reshape((self.rows*self.cols,))
        for ob in self.obstacles:
            ob.score=np.mean(confidence_pt_data[ob.grids])
            ob.height=np.mean(height_pt_data[ob.grids])

    def Classify(self,outblobs):
        classify_pt=outblobs['class_score'].reshape((5,self.rows*self.cols))
        for ob in self.obstacles:
            for k in range(5):
                ob.meta_type_probs[k]=np.mean(classify_pt[k][ob.grids])
            ob.meta_type=ob.meta_type_probs.argmax()

    def GetObjects(self,confidence_thresh=0.1,
                      height_thresh=0.5,min_pts_num=3):
        point2grid=self.vldpc.idx_row*self.cols+self.vldpc.idx_col # 在map中的索引
        ind=range(point2grid.size)        
        # 筛选点云数据
        for point,z,i in zip(point2grid,self.vldpc.valid_z,ind):
            obs_id=self.id_img[point]
            if obs_id>=0 and self.obstacles[obs_id].score>=confidence_thresh:
                if height_thresh<0 or z<=self.obstacles[obs_id].height+height_thresh:
                    self.obstacles[obs_id].cloud.append(i)
        for ob in self.obstacles:
            j=0
            if len(ob.cloud)<min_pts_num:
                continue
            self.objects.append(obj.Objects())
            self.objects[j].cloud=ob.cloud
            self.objects[j].score=ob.score
            self.objects[j].type=self.GetObjectType(ob.meta_type)
            self.objects[j].type_probs=self.GetObjectTypeProbs(ob.meta_type_probs)
            j+=1


    def GetObjectType(self,meta_type_id):
        if meta_type_id==0:
            return 'Unknown'
        elif meta_type_id==1 \
              or meta_type_id==2:
            return 'Vehicle'
        elif meta_type_id==3:
            return 'Bicycle'
        else:
            return 'Pedestrian'

    @staticmethod
    def GetObjectTypeProbs(meta_type_probs):
        object_type_probs=np.zeros(4)
        object_type_probs[0]=meta_type_probs[0]
        object_type_probs[1]=meta_type_probs[1]+meta_type_probs[2]
        object_type_probs[2]=meta_type_probs[3]
        object_type_probs[3]=meta_type_probs[4]
        return object_type_probs

