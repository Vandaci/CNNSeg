# -*- coding: utf-8 -*-
import numpy as np  

class FeatureGenerator():
    def __init__(self,validpoints):
        self.pointcloud=validpoints
        self.feature_blob=None
        self.rows=self.pointcloud.rows
        self.cols=self.pointcloud.cols
        self.grids=self.rows*self.cols
        self.lrange=self.pointcloud.lrange
        self.__log_table=np.log1p(np.arange(256))

    def Generate(self):
        x=self.pointcloud.valid_x
        y=self.pointcloud.valid_y
        z=self.pointcloud.valid_z
        intensity=self.pointcloud.valid_intensity
        idx_row=self.F2I(x,self.rows,self.lrange)
        idx_col=self.F2I(y,self.cols,self.lrange)
        map_idx=idx_row*self.pointcloud.cols+idx_col
        uidx,counts=np.unique(map_idx,return_counts=True)
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
        self.feature_blob=np.concatenate([max_height_data.reshape((1,1,self.rows,self.cols)),
                                      mean_height_data.reshape((1,1,self.rows,self.cols)),
                                      grid_count.reshape((1,1,self.rows,self.cols)),
                                      direction_data.reshape((1,1,self.rows,self.cols)),
                                      top_intensity_data.reshape((1,1,self.rows,self.cols)),
                                      mean_intensity_data.reshape((1,1,self.rows,self.cols)),
                                      distance_data.reshape((1,1,self.rows,self.cols)),
                                      none_empty.reshape((1,1,self.rows,self.cols))],axis=1)
        return self.feature_blob

    def LogCount(self,count_data):
        if count_data<256:
            return self.__log_table[count_data]
        return np.log(1+count_data)

    @staticmethod
    def F2I(x,rows,lrange):
        return np.floor(rows*(lrange-x)/(2*lrange)).astype(np.int32)
    @staticmethod
    def Pix2Pc(in_pixel,in_size,out_range):
        res=2.0*out_range/in_size
        return out_range-(in_pixel+0.5)*res


if __name__=='__main__':
    import pointcloud as pc 
    import validpoint
    tst=pc.PointCloud()
    tst.ReadFromBinFile('/home/reme/桌面/CNNSeg/data/test.pcd')
    vldp=validpoint.ValidPoints(tst,640,640,5,-5,60)
    fg=FeatureGenerator(vldp)
    fg.Generate()
    