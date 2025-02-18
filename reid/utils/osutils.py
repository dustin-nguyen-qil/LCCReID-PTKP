from __future__ import absolute_import
import os
import errno
import time 

def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def time_now():
    '''return current time in format of 2000-01-01 12:01:01'''
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())