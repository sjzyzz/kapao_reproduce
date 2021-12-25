"""
    this python file will change `sys.path` so that I can use some modules in tools dir,
    the modules contains:
        /home/sjzyzz/kapao
        /home/sjzyzz/kapao_reproduce/lib(in relative form which mean i can use in whatever prefix, i hope you get my point)
"""

# TODO: what the hell is this fucking code doing lol
#       but it is kind of boring, 
#       and i have not encounted that situation
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
lib_parent_dir = osp.join(this_dir, '..')

add_path('/home/sjzyzz/kapao')
add_path(lib_parent_dir)
# print(sys.path)
# import kapao
