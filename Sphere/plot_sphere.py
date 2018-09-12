# Plot the results from the 2D pose graph optimization.
# It will draw a line between consecutive vertices, the command line expects two optional file names:
#
#   ./plot_sphere.py --initial_poses file1 --optimized_poses file2
#
# The initial pose and optimized pose file have the following format:
#   ID x y z q_x q_y q_z q_w

import numpy as np
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
from optparse import OptionParser

parser = OptionParser()
parser.add_option('--initial_poses', dest='initial_poses',
                  default='../cmake-build-release/bin/sphere_original.txt',
                  help='the filename that contains the original poses')
parser.add_option('--optimized_poses', dest='optimized_poses',
                  default='../cmake-build-release/bin/sphere_optimized.txt',
                  help='the filename that contains the optimized poses')
(options, args) = parser.parse_args()

# read the original and optimized poses files
pose_original = None
if options.initial_poses != '':
    pose_original = np.genfromtxt(options.initial_poses, usecols=(1, 2, 3))
pose_optimized = None
if options.optimized_poses != '':
    pose_optimized = np.genfromtxt(options.optimized_poses, usecols=(1, 2, 3))

# plot the results in XY plane
fig = plot.figure('Pose Graph 3D')
if pose_original is not None:
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(pose_original[:, 0], pose_original[:, 1], pose_original[:, 2], '-', color='green')
    ax.set_aspect('equal')
    ax.set_title('Origin Pose Graph 3D')

if pose_optimized is not None:
    ax = fig.add_subplot(122, projection='3d')
    ax.plot(pose_optimized[:, 0], pose_optimized[:, 1], pose_optimized[:, 2], '-', color='red')
    ax.set_aspect('equal')
    ax.set_title('Optimized Pose Graph 3D')

plot.show()
