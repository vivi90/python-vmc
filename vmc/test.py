#!/usr/bin/env python3

from log import Log
import sys
from vrm import VRM as Model
from vmc import Assistant as VMCAssistant, Quaternion
from gui import Window

# Configuration
connection = {
    "host" : "localhost",
    "port" : 39539,
    "name" : "example"
}

# Logging
sys.stdout = Log(filename = "vmc.log", is_error = False)
sys.stderr = Log(filename = "vmc.log", is_error = True)

# Avatar model
model = Model("test.vrm")
print(model.get_bones()[101]) # leftUpperLeg (see: https://github.com/vrm-c/vrm-specification/issues/371)
model_root = []
model_t_pose = []

# Quaternions example
q = Quaternion.from_euler(-90, -180, 90, precision = 12)
print(q)
e = q.to_euler()
print(e)

# GUI
Window(
    VMCAssistant(
        connection['host'],
        connection['port'],
        connection['name']
    ),
    model_root,
    model_t_pose
).MainLoop()
