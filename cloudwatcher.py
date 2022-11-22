
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
from math import atan2,pi

def calculate_cloud_movement(could1,cloud2,downscale_amount=1, show_visual = True,save_visual_path=None):
    downscale = downscale_amount
    prvs = cv2.cvtColor(cloud1, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(cloud2, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #0.5 image scale
    #3number of pyramide layers
    #15 window size computation
    #3 number iterations
    #5 polynomial degree expansion
    #1.2 standard deviation
    u = flow[..., 0]
    v = -flow[..., 1]

    u_downscaled=cv2.resize(u,(int(u.shape[1]/downscale),int(u.shape[0]/downscale)))
    v_downscaled=cv2.resize(v,(int(v.shape[1]/downscale),int(v.shape[0]/downscale)))
    im_downscaled= cv2.resize(cloud1,(int(v.shape[1]/downscale),int(v.shape[0]/downscale)))
    
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mag_downscaled = cv2.resize(mag,(int(v.shape[1]/downscale),int(v.shape[0]/downscale)))

    #vects = np.vstack((u.flatten(),v.flatten()))
    vects = np.vstack((u_downscaled.flatten(), v_downscaled.flatten()))

    avg_vector_length = np.linalg.norm(vects,axis=0).mean()
    #avg_x = u.mean()
    #avg_y = v.mean()
    avg_x = u_downscaled.mean()
    avg_y = v_downscaled.mean()
    avg_ang = atan2(avg_x,avg_y)*180/pi

    print(u_downscaled, "and", v_downscaled,"mmmm", vects)

    if show_visual or save_visual_path:
        f=plt.figure()
        plt.imshow(im_downscaled)
        qq=plt.quiver(np.arange(u_downscaled.shape[1]),np.arange(u_downscaled.shape[0]),u_downscaled,v_downscaled,mag_downscaled,cmap=plt.cm.jet)
        plt.colorbar(qq, cmap=plt.cm.jet)
        if show_visual:
            plt.show()
        if save_visual_path is not None:
            plt.savefig(save_visual_path)
    return u,v,avg_vector_length, avg_x,avg_y,avg_ang
    
    

if __name__ == '__main__':
    args = sys.argv[1:]
    im1 = args[0]
    im2 = args[1]
    cloud1 = cv2.imread(im1)
    cloud2 = cv2.imread(im2)
    scale = 1
    vis = False
    if len(args) > 2 :
        scale = float(args[2])
    if "-v" in args:
        vis = True

    u,v,avg_length,avg_x,avg_y,avg_ang = calculate_cloud_movement(cloud1,cloud2,downscale_amount=scale,show_visual=vis)
    print('average movement: ', avg_length ,' pixels  at ',avg_ang,' degrees')

