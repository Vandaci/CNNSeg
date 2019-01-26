# CNNSeg
CNNSeg Algorithm in APOLLO3.0
* 这是Apollo3.0 Lidar CNNSeg版本，出于无奈调试不了，c++由不熟，且Caffe Python接口较为简单，因此才用Python，以加深算法印象与理解  
## 1.文件结构说明
* ``.vscode/   `` 
配置Python解释器，我使用的是Anaconda安装的caffe，其中利用了别人的第三方库``PyPCL``读取点云pcd数据，感谢！！！单读取必要的信息应该较为简单，后续考虑自己写。  
[Anaconda安装caffe教程](https://github.com/Vandaci/accumulation_of_apollo/blob/master/how_to_install_conda_on_ubuntu.md)
    ```
    $ pip install PyPCL
    ```
* ``data/``     
  存储本仿真示例所用到点云数据
  $ \Rightarrow $ ``'17.bin'`` 
        点云二进制数据，来源Apollo数据开放平台，标注样例
  $ \Rightarrow $ ``‘test.pcd’``
        单帧点云数据文件，来源于Apollo3.0
* ``model/   ``
    存储caffe框架下cnnseg预先训练的``deploy.caffemodel``模型文件，``deploy.prototxt``为定义网络的配置文件。可以才用下面的方法初始化网络模型：  

    ```
    import caffe 
    net=caffe.Net('deploy.prototxt','deploy.caffemodel',caffe.TEST)
    ```
* ``notebook/   ``
  存储由jupyter notebook开发的程序，主要方便数据可视化,jupyter notebook常见的一些问题及安装方式见我github的另一个软件仓库，``Apollo日积月累``  
* ``old version/   ``
  存储程序迭代过程中的旧版本  
* ``pictures``   
  存储cnnseg网络框架图，生成方法：
  ```
  $ draw_net deploy.prototxt net.png
  # draw_net 网络文件.prototxt *.png
  ```
  生成网络图需要配置好caffe框架，建议使用conda安装，安装方法同见另一个仓库    
* ``tmp/ ``   
  临时文件夹，主要存储的临时的Python试语法文件，或者临时测试算法的正确性脚本  
* ``src/``    
  CNNSeg源代码文件夹，cnnseg各模块算法  
  $ \Rightarrow $ ``cluster2d.py`` 簇类，主要应用联合查找算法，用于障碍物聚类    
  $ \Rightarrow $``featuregenerator``特征生成类，用于处理点云``*.pcd``文件，生成cnnseg网络需要的数据  
  $ \Rightarrow $ ``cnnsegmentation.py`` 
    cnnseg类，生成一个cnnseg对象，整合cnnseg所有过程 
  $ \Rightarrow $``node.py``
    节点类，主要用于cluster实现，用于障碍物聚类
  $ \Rightarrow $ ``obstacles.py``
    障碍物类，存储聚类后的障碍物对象
  $ \Rightarrow $ ``objects.py``
    对象类，用于存储一个cnnseg过程的障碍物对象
  $ \Rightarrow $ ``util.py``
    用于setfind算法的一些实现
  $ \Rightarrow $ ``validpoint``
    有效点云类，从原始点云数据中分离出感兴趣的点云
* ``docs/``
    存储本算法的一些说明文档与帮助文档

## 2.目录  

### 2.1源代码索引

   [cluster2d](src/cluster2d.py)
   [cnnsegmentation](src/cnnsegmentation.py)
   [featuregenerate](src/featuregenerate.py)
   [node](src/node.py)
   [objects](src/objects.py)
   [obstacle](src/obstalce.py)
   [pointcloud](src/pointcloud.py)
   [util](src/util.py)
   [validpoint](src/validpoint.py)








