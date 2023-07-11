'''
https://developer.blender.org/rB24e01654630#change-Z2YKyxIGBVqs

'''


import math


from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

import numpy as np
import cv2
#
# Copyright 2011-2021 Blender Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# <pep8 compliant>

# Fit to match default projective camera with focal_length 50 and sensor_width 36.
default_fisheye_polynomial = [-1.1735143712967577e-05,
                              -0.019988736953434998,
                              -3.3525322965709175e-06,
                              3.099275275886036e-06,
                              -2.6064646454854524e-08]

# Utilities to generate lens polynomials to match built-in camera types, only here
# for reference at the moment, not used by the code.
def create_grid(sensor_height, sensor_width):
    import numpy as np
    if sensor_height is None:
        sensor_height = sensor_width / (16 / 9)  # Default aspect ration 16:9
    uu, vv = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    uu = (uu - 0.5) * sensor_width
    vv = (vv - 0.5) * sensor_height
    rd = np.sqrt(uu ** 2 + vv ** 2)
    
    #fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    #ax.plot_surface(uu, vv, rr)
    #plt.show()

    # Plot the surface.

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(uu, vv, rd, cmap=cm.coolwarm,
                        linewidth=0.1, antialiased=False)
    ax.grid(color='r', linestyle='-', linewidth=1)
    # Customize the z axis plot.
    ax.set_zlim(0, rd.max())
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(azim=0, elev=90)
    plt.show()


    return rd
def draw_distortion(image_width, image_height):
    distortion = np.array([0.3, 0.001, 0.0, 0.0, 0.01])
    # Generate Grid of Object Points
    grid_size = [100, 100]
    square_size = 0.1
    
    uu, vv = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    uu = (uu - 0.5) * image_width
    vv = (vv - 0.5) * image_height
    rd = np.sqrt(uu ** 2 + vv ** 2)

    object_points = np.stack([uu, vv])

   # mx, my = [(grid_size[0] - 1) * square_size / 2, (grid_size[1] - 1) * square_size / 2]
   # for i in range(grid_size[0]):
    #    for j in range(grid_size[1]):
   #         object_points[i * grid_size[0] + j] = [i * square_size - mx, j * square_size - my, 0]

    # Setup the camera information
    f, p = [5e-3, 120e-8]
    intrinsic = np.array([[f/p, 0, 0], [0, f/p, 0], [0, 0, 1]])
    rvec = np.array([0.0, 0.0, 0.0])
    tvec = np.array([0.0, 0.0, 3.0])

    # Project the points
    image_points, jacobian = cv2.projectPoints(object_points, rvec, tvec, intrinsic, distortion)

    # Plot the points (using PyPlot)
    plt.scatter(*zip(*image_points[:, 0, :]), marker='.')
    plt.axis('equal')
    plt.xlim((uu.min(), uu.max()))
    plt.ylim((vv.min(), vv.max()))
    plt.grid()
    plt.show()


# Ref: Tutorial: https://www.statology.org/curve-fitting-python/
#define function to calculate adjusted r-squared
# value tells us the percentage of the variation in the response variable
def adjR(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results['r_squared'] = 1- (((1-(ssreg/sstot))*(len(y)-1))/(len(y)-degree-1))

    return results

    # Model default used
def fisheye_lens_polynomial_from_distorted_projective_polynomial(k1, k2, k3, focal_length=50, sensor_width=36, sensor_height=None):
    import numpy as np
    rd = create_grid(sensor_height, sensor_width)
    r2 = (rd / focal_length) ** 2
    r4 = r2 * r2
    r6 = r4 * r2
    r_coeff = 1 + k1 * r2 + k2 * r4 + k3 * r6
    focal_length_pixels = focal_length * r_coeff
    
    #polynomial1 = np.polyfit(rd.flat, (-np.arctan(rd / focal_length * r_coeff)).flat, 1)
    #polynomial2 = np.polyfit(rd.flat, (-np.arctan(rd / focal_length * r_coeff)).flat, 2)
    #polynomial3 = np.polyfit(rd.flat, (-np.arctan(rd / focal_length * r_coeff)).flat, 3)
    polynomial4 = np.polyfit(rd.flat, (-np.arctan(rd / focal_length_pixels)).flat, 4)
    #polynomial5 = np.polyfit(rd.flat, (-np.arctan(rd / focal_length * r_coeff)).flat, 5)
    #polynomial6 = np.polyfit(rd.flat, (-np.arctan(rd / focal_length * r_coeff)).flat, 6)

    #model1 = np.poly1d(polynomial1)
    #model2 = np.poly1d(polynomial2)
    #model3 = np.poly1d(polynomial3)
    model4 = np.poly1d(polynomial4)
    #model5 = np.poly1d(polynomial5)
    #model6 = np.poly1d(polynomial6)

    polyline = np.linspace(0, rd.max(), 1000)

    plt.scatter(rd.flat, (-np.arctan(rd / focal_length_pixels )).flat)

    #add fitted polynomial lines to scatterplot 
    #plt.plot(polyline, model1(polyline), color='green')
    #plt.plot(polyline, model2(polyline), color='red')
    #plt.plot(polyline, model3(polyline), color='purple')
    plt.plot(polyline, model4(polyline), color='blue')
    #plt.plot(polyline, model5(polyline), color='orange')
    #plt.plot(polyline, model6(polyline), color='yellow')
    #plt.show()

    print(model4)
    for i in range(6):
        print(adjR(rd.flat, (-np.arctan(rd / focal_length_pixels)).flat, i))

    return list(reversed(polynomial4))

def fisheye_lens_polynomial_from_projective(focal_length=50, sensor_width=36, sensor_height=None):
    import numpy as np
    rr = create_grid(sensor_height, sensor_width)
    polynomial = np.polyfit(rr.flat, (-np.arctan(rr / focal_length)).flat, 4)
    return list(reversed(polynomial))


def fisheye_lens_polynomial_from_projective_fov(fov, sensor_width=36, sensor_height=None):
    import numpy as np
    f = sensor_width / 2 / np.tan(fov / 2)
    return fisheye_lens_polynomial_from_projective(f, sensor_width, sensor_height)


def fisheye_lens_polynomial_from_equisolid(lens=10.5, sensor_width=36, sensor_height=None):
    import numpy as np
    rr = create_grid(sensor_height, sensor_width)
    x = rr.reshape(-1)
    x = np.stack([x**i for i in [1, 2, 3, 4]])
    y = (-2 * np.arcsin(rr / (2 * lens))).reshape(-1)
    polynomial = np.linalg.lstsq(x.T, y.T, rcond=None)[0]
    return [0] + list(polynomial)


def fisheye_lens_polynomial_from_equidistant(fov=180, sensor_width=36, sensor_height=None):
    import numpy as np
    return [0, -np.radians(fov) / sensor_width, 0, 0, 0]



def fisheye_lens_polynomial_from_distorted_projective_divisions(k1, k2, focal_length=50, sensor_width=36, sensor_height=None):
    import numpy as np
    rr = create_grid(sensor_height, sensor_width)
    r2 = (rr / focal_length) ** 2
    r4 = r2 * r2
    r_coeff = 1 + k1 * r2 + k2 * r4
    #
    polynomial = np.polyfit(rr.flat, (-np.arctan(rr / focal_length / r_coeff)).flat, 4)
    return list(reversed(polynomial))

 #Distortion coefficient:
#   [[-1.42882885e-01 -3.77996876e+00 -3.08953071e-03 -1.31954369e-03, 8.84370517e+01]]
#   -1.92587195e-01  1.06400645e+00 -2.69259175e-03 -1.13518455e-03 -3.70193794e+01
#    distortion_coefficients=(k1 k2 p1 p2 k3)

def main():
    
    # parms = fisheye_lens_polynomial_from_distorted_projective_polynomial(k1, k2, k3, focal_length=50, sensor_width=36)
    #parms = fisheye_lens_polynomial_from_projective(focal_length=30.27, sensor_width=26, sensor_height=None)
    #parms = fisheye_lens_polynomial_from_projective(focal_length=50, sensor_width=36, sensor_height=None)
        # Image sensor in mm
    sensor_width= 7.564 
    sensor_height= 5.476

    parms = fisheye_lens_polynomial_from_distorted_projective_polynomial(-9.66487717e-02, -6.96678653e+00 ,  1.25015788e+02, focal_length=21.425468071117212, sensor_width=7.564 , sensor_height=5.476 )


    # focal length in pixel from opencv calibration
    fx = 1.07195479e+04
    fy = 1.06985464e+04 
    # Raw image size
    image_width = 4056
    image_height = 3040
    # Image sensor in mm
    w = sensor_width 
    h = sensor_height
    #Fx = fx * image_width / w   # 5192pix * 23.5mm / 6000pix = 20.33mm
    #Fy = fy * image_height / h   # 5192pix * 15.6mm / 4000pix = 20.25mm
    #print("value FX", Fx)
    #print("value FY", Fy)


    # barrel distortion typically k1 > 0 and pincushion distortion typically k1 < 0
    parms_blender_gui = [round(math.degrees(x), 6) for x in parms]
    print(parms)
    print( parms_blender_gui )
    
    draw_distortion(image_width, image_height)


if __name__ == '__main__':
    main()
