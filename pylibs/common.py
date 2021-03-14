# coding: utf-8
"""
 @Author  : Shuai Liao
"""

# from basic.common import plt, plt_wait, cv2_imshow, cv2_wait, cv2_putText, generate_mp4

from __future__ import print_function
import sys
import os
import io, imageio
import cv2
import matplotlib.pyplot as plt
import platform
import re
from collections import OrderedDict as odict
from ast import literal_eval  # safe way to do eval('str_expression')
import numpy as np


os_name = platform.system()
is_mac   = (os_name=='Darwin')
is_linux = (os_name=='Linux' )

# One liner detect if it's ran by python3
is_py3 = (sys.version_info > (3, 0))

class Env(object):
    def __getattr__(self, name):
        return os.getenv(name)
env = Env()
# Example:
# from basic.common imoprt env
# print env.HOME

def argv2dict(sys_argv):
    """ Auto parsing  ---key=value like arguments into dict.
    """
    # if matches, return 2 groups: key, value # correspond to 2 ()s
    named_pattern_exp = '--(?P<key>[a-zA-Z0-9_-]*)=(?P<value>.*)'
    pattern = re.compile(named_pattern_exp)

    rslt = odict()
    for arg in sys_argv:
        match   = pattern.match(arg)
        if match:  # match is not None
            gd = match.groupdict()
            key = gd['key']
            try:# To test if 'value' string is a python expression.
                # it's a python expression, here can return float, int, list etc.
                rslt[key] = literal_eval( gd['value'] )
            except:
                # it's not a python expression, just return such string.
                rslt[key] = gd['value']
        else:
            # print('Matching failed. %s' % arg)
            print('  [argv2dict] skip: %s' % arg)
            pass
    return rslt


def mkdir4file(filename):
    try:
        os.makedirs(os.path.split(filename)[0])
    except:
        pass

def Open(filepath, mode='r'): #, overwrite=1
    """ wrapper of open, auto make dirs when open new write file. """
    if mode.startswith('w') or mode.startswith('a'):
        try:
            os.makedirs( os.path.dirname(filepath) )
            print( "[Warning] Open func create dir: ", os.path.dirname(filepath) )
        except: pass
    return open(filepath, mode)




# Dynamically add PYTHONPATH for import
def add_path(*paths):
    for path in paths:
        path = os.path.abspath(path)
        assert os.path.exists(path), "[Warning] path not exist: %s" % path
        sys.path.insert(0, path)
        #if path not in sys.path:
        #    sys.path.insert(0, path)




def quit_figure(event):
    # print("event.key", event.key)
    if   event.key == 'q':
        plt.close(event.canvas.figure)
    elif event.key == 'escape':
        plt.close(event.canvas.figure)
        exit()
#
def plt_wait(delay=None, event_func=quit_figure): # delay: in millisecond.
    if delay is not None:
        plt.tight_layout()
        plt.draw() #
        plt.ioff()
        plt.waitforbuttonpress(delay/1000) # 1e-4) # 0.00001)
    else:
        cid = plt.gcf().canvas.mpl_connect('key_press_event', event_func)
        plt.tight_layout()
        plt.show() #(block=False)

def plt_save(file_name, dpi=None, transparent=False, bbox_inches='tight', pad_inches=0, **kwargs):
    plt.tight_layout()
    mkdir4file(file_name)
    # plt.savefig(file_name, dpi=None, transparent=transparent, **kwargs)
    plt.savefig(file_name, dpi=None, transparent=transparent, bbox_inches=bbox_inches, pad_inches=pad_inches, **kwargs)




def cv2_wait(delay=-1):  # delay: in millisecond.
    key = cv2.waitKey(delay) & 0xFF
    if key==27:          # Esc key to stop
        cv2.destroyAllWindows()
        exit()
    return key

def cv2_putText(image, left_bottom, display_txt, bgcolor=None, fgcolor=(0,0,225), scale=0.5,thickness=1,line_space=1.8):
    """
    line_space: a line to occupy xline_space time of actual line height.
    """
    xmin, ymax = left_bottom  # Note: here xmin, ymax is of the text box
    fontface, fontscale, color, thickness = cv2.FONT_HERSHEY_DUPLEX, scale, fgcolor, thickness
    # image[:] = np.ascontiguousarray(image)  # since opencv 4.2.0,  even if np.copy(image) doesn't work.
    assert image.flags['C_CONTIGUOUS'], "Please call image = np.ascontiguousarray(image)  before this function." # Since opencv 4.2.0,  even if np.copy(image) doesn't work.
    if display_txt.find('\n')>=0:
        size = cv2.getTextSize(display_txt, fontface, fontscale, thickness)[0]
        lines = display_txt.split('\n')
        for i,line_txt in enumerate(lines):
            image = cv2_putText(image, (xmin, int(ymax+i*size[1]*line_space)), line_txt, bgcolor=bgcolor, fgcolor=fgcolor, scale=scale,thickness=thickness)
        return image
    else:
        if bgcolor is not None:
            size = cv2.getTextSize(display_txt, fontface, fontscale, thickness)[0]
            top_left, bottom_right = (xmin, ymax-size[1]), (xmin+size[0], ymax+5) # for ymin and 5 more pixel
            cv2.rectangle(image, top_left, bottom_right, bgcolor, cv2.FILLED)     # opencv2 CV_FILLED
        cv2.putText(image, display_txt, (xmin, ymax), fontface, fontscale, color,thickness,lineType=cv2.LINE_AA)
        return image


def generate_animation(out_file, frames, rsz_height=None, duration=1):
    # resize frames
    img = frames[0]
    h,w = img.shape[:2]
    #
    Side = rsz_height
    if Side is None or Side<=0:
        scale=1.0
    else:
        scale = Side/float(h) # (maxside)
    rsz_frames = [cv2.resize(frame, None, fx=scale,fy=scale) for frame in frames]

    mkdir4file(out_file)
    _, ext = os.path.splitext(out_file)
    if   ext == '.gif':
        gif_kwargs = dict(duration=duration)  # refs: https://imageio.readthedocs.io/en/stable/format_gif-pil.html#gif-pil
        with imageio.get_writer(out_file, format='gif', mode='I', **gif_kwargs) as writer:
            for image in rsz_frames:
                writer.append_data(image)
    elif ext == '.mp4':
        with imageio.get_writer(out_file, fps=len(rsz_frames)/duration) as writer:
            for image in rsz_frames:
                writer.append_data(image)
    else:
        raise NotImplementedError(f'Unknown output format: {ext}')

