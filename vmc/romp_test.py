#!/usr/bin/env python3

from log import Log
import sys
from configuration import Configuration
from vmc import Assistant as VMCAssistant, Bone, Position, Quaternion, Timestamp
import numpy as np
import romp, cv2
from scipy.spatial.transform import Rotation as r

# Configuration
configuration: dict = {
    "host"  : "localhost",
    "port"  : 39539,
    "name"  : "example",
    "delta" : 0.0
}

# Logging
sys.stdout = Log(filename = "vmc.log", is_error = False)
sys.stderr = Log(filename = "vmc.log", is_error = True)

# ROMP
settings = romp.main.default_settings
subject_id = 0
romp_model = romp.ROMP(settings)
outputs = romp_model(cv2.imread('pose.png'))
smpl_root_position = outputs["cam_trans"][subject_id]
smpl_rotations_by_axis = outputs["smpl_thetas"][subject_id].reshape(24, 3)

# Info
rotation_by_axis = smpl_rotations_by_axis[0] # Pelvis
rotation_by_quaternion = r.from_rotvec(rotation_by_axis).as_quat()
print("ROMP settings: " + str(settings))
print("Root position: " + str(smpl_root_position))
print("Pelvis rotation by axis: " + str(rotation_by_axis))
print("Pelvis rotation by quaternion: " + str(rotation_by_quaternion))

# VRM
# (Just the logical bone names. Not the actual ones [nodes] in the rig)
vrm_bone_names = {
    0 : "Hips", # Pelvis
    1 : "LeftUpperLeg", # L_Hip
    2 : "RightUpperLeg", # R_Hip
    3 : "Spine", # Spine1
    4 : "LeftLowerLeg", # L_Knee
    5 : "RightLowerLeg", # R_Knee
    6 : "Chest", # Spine2
    7 : "LeftFoot", # L_Ankle
    8 : "RightFoot", # R_Ankle
    9 : "UpperChest", # Spine3
    10 : "LeftToes", # L_Foot
    11 : "RightToes", # R_Foot
    12 : "Neck", # Neck
    13 : "LeftShoulder", # L_Collar
    14 : "RightShoulder", # R_Collar
    15 : "Head", # Head
    16 : "LeftUpperArm", # L_Shoulder
    17 : "RightUpperArm", # R_Shoulder
    18 : "LeftLowerArm", # L_Elbow
    19 : "RightLowerArm", # R_Elbow
    20 : "LeftHand", # L_Wrist
    21 : "RightHand", # R_Wrist
    22 : "LeftMiddleProximal", # L_Hand
    23 : "RightMiddleProximal" # R_hand
}

vrm_swapped_bone_names = {
    0 : "Hips", # Pelvis
    1 : "LeftUpperLeg", # L_Hip
    2 : "RightUpperLeg", # R_Hip
    3 : "Spine", # Spine1
    4 : "LeftLowerLeg", # L_Knee
    5 : "RightLowerLeg", # R_Knee
    6 : "Chest", # Spine2
    7 : "LeftFoot", # L_Ankle
    8 : "RightFoot", # R_Ankle
    9 : "UpperChest", # Spine3
    10 : "LeftToes", # L_Foot
    11 : "RightToes", # R_Foot
    12 : "Neck", # Neck
    14 : "LeftShoulder", # L_Collar
    13 : "RightShoulder", # R_Collar
    15 : "Head", # Head
    17 : "LeftUpperArm", # L_Shoulder
    16 : "RightUpperArm", # R_Shoulder
    19 : "LeftLowerArm", # L_Elbow
    18 : "RightLowerArm", # R_Elbow
    21 : "LeftHand", # L_Wrist
    20 : "RightHand", # R_Wrist
    23 : "LeftMiddleProximal", # L_Hand
    22 : "RightMiddleProximal" # R_hand
}

"""
# Swap bones (wrong)
vrm_bone_names = vrm_swapped_bone_names
"""

# VMC
configuration = Configuration("vmc.yml", configuration)
vmc = VMCAssistant(
    configuration['host'], 
    configuration["port"], 
    configuration["name"]
)
started_at = Timestamp()
bones = []
for index, rotation in enumerate(smpl_rotations_by_axis):
    bone_name = vrm_bone_names[index]
    rotation = r.from_rotvec(rotation).as_quat()
    rotation = Quaternion(rotation[0], rotation[1], rotation[2],  rotation[3])
    """
    # Calculate additional rotation of some bones (y-90 degrees) (Wrong)
    if bone_name in (#"LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand", "LeftMiddleProximal",
                     "RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand", "RightMiddleProximal"):
        rotation = r.from_quat([rotation.x, rotation.y, rotation.z, rotation.w])
        rotation = rotation.as_euler('xyz', degrees=True)
        print(rotation)
        rotation = r.from_euler('xyz', [rotation[0], rotation[1]-90, rotation[2]], degrees=True)
        print(rotation.as_euler('xyz', degrees=True))
        rotation = rotation.as_quat()
        rotation = Quaternion(rotation[0], rotation[1], rotation[2],  rotation[3])
        #rotation = rotation.multiply_by(Quaternion.from_euler(0, -90, 0, 12), 12)
    """
    bones.append(
        [
            Bone(bone_name),
            Position(0.0, 0.0, 0.0),
            rotation
        ]
    )

bones += [
    [Bone("LeftEye"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("RightEye"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("Jaw"), Position(0.0, 0.0, 0.0),  Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("LeftThumbProximal"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("LeftThumbIntermediate"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("LeftThumbDistal"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("LeftIndexProximal"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("LeftIndexIntermediate"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("LeftIndexDistal"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
  # [Bone("LeftMiddleProximal"), Position(0.0, 0.0, 0.0),  Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("LeftMiddleIntermediate"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("LeftMiddleDistal"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("LeftRingProximal"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("LeftRingIntermediate"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("LeftRingDistal"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("LeftLittleProximal"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("LeftLittleIntermediate"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("LeftLittleDistal"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("RightThumbProximal"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("RightThumbIntermediate"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("RightThumbDistal"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("RightIndexProximal"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("RightIndexIntermediate"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("RightIndexDistal"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
  # [Bone("RightMiddleProximal"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("RightMiddleIntermediate"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("RightMiddleDistal"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("RightRingProximal"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("RightRingIntermediate"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("RightRingDistal"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("RightLittleProximal"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("RightLittleIntermediate"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)],
    [Bone("RightLittleDistal"), Position(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)]
]
"""
# Apply correctly roatated values by y-90 degrees (wrong)
bones[14][2] = Quaternion(x=-0.038144, y=0.828361, z=0.05417, w=0.556263) # RightShoulder / R_Collar: -7.18492d, 112.225d, 0.443739d
bones[17][2] = Quaternion(x=-0.248151, y=0.863098, z=0.131794, w=0.419659) # RightUpperArm / R_Shoulder: -1.79611d, 127.831d, 31.2018d
bones[19][2] = Quaternion(x=0.433544, y=0.602222, z=0.266302, w=0.615184) # RightLowerArm / R_Elbow: -83.239d, 149.333d, -98.8831d
bones[21][2] = Quaternion(x=0.040907, y=0.772052, z=-0.019798, w=0.633933) # RightHand / R_Wrist: -6.21696d, 101.34d, -11.1615d
"""
# Sending
vmc.send_root_transform(
    Position(smpl_root_position[0], smpl_root_position[1], smpl_root_position[2]),
    bones[0][2].multiply_by(Quaternion.from_euler(0, 90, 0, 12), 12) # Hips / Pelvis rotation y+90 degree
)
vmc.send_bones_transform(bones)
vmc.send_available_states(1)
delta = started_at.delta(configuration["delta"])
vmc.send_relative_time(delta)
configuration["delta"] = delta
