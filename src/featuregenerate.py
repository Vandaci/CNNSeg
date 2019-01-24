# # -*- coding: utf-8 -*-
import pointcloud as pc 
import numpy as np  

class FeatureGenerator():
    def __init__(self,pcdata,rows,cols,lrange):
        # step 1 : load pcdata and init
        self.pointcloud=pcdata
        self.rows=rows
        self.cols=cols
        self.lrange=lrange
        self.grids=rows*cols

    def Generate(self,max_heigth,min_height):
        # step 2 : remove invalid points
        # remove outrange of height
        valid_idx=np.any([self.pointcloud.z>=min_height,
                          self.pointcloud.z<=max_heigth],axis=0)
        x=self.PullValidPoints(self.pointcloud.x,valid_idx)
        y=self.PullValidPoints(self.pointcloud.y,valid_idx)
        z=self.PullValidPoints(self.pointcloud.z,valid_idx)
        intensity=self.PullValidPoints(self.pointcloud.intensity,valid_idx)
        # step 3: transform to map_idx  
        idx_row=self.F2I(x,self.rows,self.lrange)
        idx_col=self.F2I(y,self.cols,self.lrange)
        # remove outrange of grid
        valid_idx=np.all([idx_row>=0,idx_row<self.rows,
                          idx_col>=0,idx_col<self.cols],axis=0)
        x=self.PullValidPoints(x,valid_idx)
        y=self.PullValidPoints(y,valid_idx)
        z=self.PullValidPoints(z,valid_idx)
        self.__log_table=np.log1p(np.arange(256))     #   ln(1+x)   
        intensity=self.PullValidPoints(intensity,valid_idx)
        idx_row=self.PullValidPoints(idx_row,valid_idx)
        idx_col=self.PullValidPoints(idx_col,valid_idx)
        map_idx=idx_row*self.cols+idx_col
        uidx,counts=np.unique(map_idx,return_counts=True)
        # step 4 : pre-define 8-channels
        max_height_data=np.zeros(self.grids)
        mean_height_data=np.zeros(self.grids)
        grid_count=np.zeros(self.grids)
        top_intensity_data=np.zeros(self.grids)
        mean_intensity_data=np.zeros(self.grids)
        none_empty=np.zeros(self.grids)
        for idx,count in zip(uidx,counts):
            max_height_data[idx]=np.max(z[idx==map_idx])
            mean_height_data[idx]=np.mean(z[idx==map_idx])
            top_intensity_data[idx]=np.max(intensity[idx==map_idx])/255
            mean_intensity_data[idx]=np.mean(intensity[idx==map_idx])/255
            grid_count[idx]=self.LogCount(count.astype(np.int))
        none_empty[grid_count>0]=1
        grid_col,grid_row=np.meshgrid(range(self.cols),range(self.rows))
        center_x=self.Pix2Pc(grid_row,self.rows,self.lrange)
        center_y=self.Pix2Pc(grid_col,self.cols,self.lrange)
        direction_data=np.arctan2(center_y,center_x)/(2*np.pi) # Normalized
        distance_data=np.hypot(center_x,center_y)/60-0.5
        self.out_blob=np.concatenate([max_height_data.reshape((1,1,self.rows,self.cols)),
                                      mean_height_data.reshape((1,1,self.rows,self.cols)),
                                      grid_count.reshape((1,1,self.rows,self.cols)),
                                      direction_data.reshape((1,1,self.rows,self.cols)),
                                      top_intensity_data.reshape((1,1,self.rows,self.cols)),
                                      mean_intensity_data.reshape((1,1,self.rows,self.cols)),
                                      distance_data.reshape((1,1,self.rows,self.cols)),
                                      none_empty.reshape((1,1,self.rows,self.cols))],axis=1)
        return self.out_blob

    @staticmethod
    def PullValidPoints(data,idx):
        return data[idx]
    @staticmethod
    def F2I(x,rows,lrange):
        return np.floor(rows*(lrange-x)/(2*lrange)).astype(np.int32)
    @staticmethod
    def Pix2Pc(in_pixel,in_size,out_range):
        res=2.0*out_range/in_size
        return out_range-(in_pixel+0.5)*res
    
    def LogCount(self,count_data):
        if count_data<256:
            return self.__log_table[count_data]
        return np.log(1+count_data)


if __name__=='__main__':
    test_pcd=pc.PointCloud()
    test_pcd.ReadFromBinFile('/home/reme/桌面/CNNSeg/data/test.pcd')
    fg=FeatureGenerator(test_pcd,640,640,60)
    fg.Generate(5,-5)
