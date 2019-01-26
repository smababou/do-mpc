# quaternion auxiliry functions

import numpy as NP
from casadi import *
import pdb
def quaternionFromGyr(gyr, rate):
    gyrNorm = norm_2(gyr)
    axis = gyr/gyrNorm
    angle = gyrNorm/rate
    quat_curr_prev = vertcat(cos(angle/2.0), axis*sin(angle/2.0))
    return vertcat(quat_curr_prev)

def quaternionInvert (q_):
    # if (size(q, 2) ~= 4):
        # error('input has to be Nx4')
    qInvert = vertcat(q_[0], -q_[1:])
    return qInvert

def quaternionMultiply (q1,q2):
    # q3 = DM([0,0,0,0]);
    q30 = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    q31 = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    q32 = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
    q33 = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
    q3 = vertcat(q30,q31,q32,q33)
    return q3

def quaternionRotate (q, vec):
   # % This function will rotate the vectors v (1x3 or Nx3) by the quaternions
   # % q (1x4 or Nx4)
   # % Result: q * [0,v] * q'
   # % The result will always be a vector (Nx3)

   qInv = quaternionInvert(q)
   qv = quaternionMultiply(quaternionMultiply(q, vertcat(0.0, vec)), qInv)
   v = qv[1:]
   # pdb.set_trace()
   return v
