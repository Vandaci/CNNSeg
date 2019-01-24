# -*- coding:utf-8 -*-

class Obstacle():
    def __init__(self):
        self.grids=[]
        self.cloud=None
        self.score=0.
        self.height=-5.
        self.MetaType={'META_UNKNOWN':0,
                       'META_SMALLMOT':1,
                       'META_BIGMOT':2,
                       'META_NOMOT':3,
                       'META_PEDESTRAIN':4,
                       'MAX_META_TYPE':5}
        self.meta_type_probs=[]
        self.meta_type=self.MetaType['META_UNKNOWN']
