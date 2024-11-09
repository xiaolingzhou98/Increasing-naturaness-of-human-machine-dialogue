# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 10:51:37 2018

@author: mochp
"""

import datetime

starttime = datetime.datetime.now()

for i in range(100):
    predict('这家酒店服务不错下次还会再来')
    
endtime = datetime.datetime.now()
print (endtime - starttime)