import tkinter as tk
#import time
import dynamics as dyn
import numpy as np

window = tk.Tk()
window.title('Vehicle Animation')

side_length = 800

canvas = tk.Canvas(window, width=side_length, height=side_length, bg = 'white')
#canvas.pack()
canvas.grid(row=1, column=0)

time_label = tk.Label(text='0.0', background='black', foreground='white', width=100)
#time_label.pack(pady=20)
time_label.grid(row=0, column=0)

# The FrameRate is determined from the time step delta_t
# We use every sample so fps = 1/delta_t

# 1 meter = 10 pixels -> scale = 10
scale = 10
wheel_l = 1.2

vehicle = canvas.create_line(0,0,-10,10,width=2,fill='black')
wheel_r = canvas.create_line(0,0,-10,10,width=5,fill='red')
wheel_f = canvas.create_line(0,0,-10,10,width=5,fill='blue')
com = canvas.create_oval(0,0,10,-10, fill = 'black')
#velvec = canvas.create_line(0,0,-10,10,width=2,arrow=tk.LAST)

'''
# Circular trajectories with the standard set of radius
R1 = 5*scale
R2 = 10*scale
R3 = 20*scale
center = canvas.create_oval(side_length/2-3,side_length/2-3,side_length/2+3,side_length/2+3, fill = 'black')
traj1 = canvas.create_oval(side_length/2-R1,side_length/2-R1,side_length/2+R1,side_length/2+R1)
traj2 = canvas.create_oval(side_length/2-R2,side_length/2-R2,side_length/2+R2,side_length/2+R2)
traj3 = canvas.create_oval(side_length/2-R3,side_length/2-R3,side_length/2+R3,side_length/2+R3)
'''

# grid
center = canvas.create_oval(side_length/2-3,side_length/2-3,side_length/2+3,side_length/2+3, fill = 'black')
square_size = 5*scale
n_tiles = int(side_length/square_size)
for i in range(n_tiles):
    canvas.create_line(0,i*square_size,side_length,i*square_size)
    canvas.create_line(i*square_size,0,i*square_size, side_length)

b = dyn.b
a = dyn.a

offset_y = side_length/2
offset_x = side_length/2

n_points = 100
points = range(-n_points-2,0)
trajectory = canvas.create_line(tuple(points),smooth=True, width=3, fill='green')

def moveVehicle(xx, uu, time_sec):

    time_label.config(text=round(time_sec,1))

    # 'Extract' animation variables
    x = xx[0]
    y = xx[1]
    psi = xx[2]
    
    delta = uu[0]

    # compute wheels and body limit points
    vehicle_start_x = x-b*np.cos(psi)
    vehicle_start_y = -y+b*np.sin(psi)
    vehicle_end_x = x+a*np.cos(psi)
    vehicle_end_y = -y-a*np.sin(psi)

    wheel_r_start_x = vehicle_start_x-wheel_l/2*np.cos(psi)
    wheel_r_start_y = vehicle_start_y+wheel_l/2*np.sin(psi)
    wheel_r_end_x = vehicle_start_x+wheel_l/2*np.cos(psi)
    wheel_r_end_y = vehicle_start_y-wheel_l/2*np.sin(psi)

    wheel_f_start_x = vehicle_end_x-wheel_l/2*np.cos(psi+delta)
    wheel_f_start_y = vehicle_end_y+wheel_l/2*np.sin(psi+delta)
    wheel_f_end_x = vehicle_end_x+wheel_l/2*np.cos(psi+delta)
    wheel_f_end_y = vehicle_end_y-wheel_l/2*np.sin(psi+delta) 

    # scale averything
    vehicle_start_x = vehicle_start_x*scale+offset_x
    vehicle_start_y = vehicle_start_y*scale+offset_y
    vehicle_end_x = vehicle_end_x*scale+offset_x
    vehicle_end_y = vehicle_end_y*scale+offset_y

    wheel_r_start_x = wheel_r_start_x*scale+offset_x
    wheel_r_start_y = wheel_r_start_y*scale+offset_y
    wheel_r_end_x = wheel_r_end_x*scale+offset_x
    wheel_r_end_y = wheel_r_end_y*scale+offset_y

    wheel_f_start_x = wheel_f_start_x*scale+offset_x
    wheel_f_start_y = wheel_f_start_y*scale+offset_y
    wheel_f_end_x = wheel_f_end_x*scale+offset_x
    wheel_f_end_y = wheel_f_end_y*scale+offset_y

    com_x1 = x*scale+offset_x-3
    com_y1 = -y*scale+offset_y-3
    com_x2 = x*scale+offset_x+3
    com_y2 = -y*scale+offset_y+3

    #velvec_x2 = com_x1 + V/2*scale*np.cos(beta + psi)
    #velvec_y2 = com_y1 - V/2*scale*np.sin(beta + psi)

    canvas.coords(vehicle,vehicle_start_x,vehicle_start_y,vehicle_end_x,vehicle_end_y)
    canvas.coords(wheel_f,wheel_f_start_x,wheel_f_start_y,wheel_f_end_x,wheel_f_end_y)
    canvas.coords(wheel_r,wheel_r_start_x,wheel_r_start_y,wheel_r_end_x,wheel_r_end_y)
    canvas.coords(com, com_x1, com_y1, com_x2, com_y2)


def drawTraj(ss):

    # Update the value of the offset such that the trajectory is centered in the canvas
    # (otherwise it may happen that part of the trajectory is out of the screen)
    mean_x = np.mean(ss[0,:])
    mean_y = np.mean(ss[1,:])
    global offset_x, offset_y
    offset_x = side_length/2 + mean_x*scale
    offset_y = side_length/2 + mean_y*scale
    
    # canvas.create_line((x1, y1, x2, y2, ..., xn, yn), smooth = True)
    # Let's create the (x1, y1, ...) vector
    # if tf = 20 and dt = 0.001 and n_points 100 -> resolution = 20/0.01/100 = 20
    # so basically we consider 1 every 200 points from the ss trajectory
    resolution = int(dyn.tf/dyn.dt/n_points)
    xxs = ss[0,::resolution]
    yys = ss[1,::resolution]
    points = np.zeros(2*n_points+2)
    points[0:-2:2] = xxs*scale+offset_x
    points[1:-1:2] = -yys*scale+offset_y
    points[-2] = ss[0,-1]*scale+offset_x
    points[-1] = -ss[1,-1]*scale+offset_y
    canvas.coords(trajectory,tuple(points))

