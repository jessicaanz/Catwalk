import mujoco
import mujoco.viewer
import time
import os

# Setup Parameters
relative_path = "cat_model.xml"
scene_path = os.path.abspath(relative_path)
model = mujoco.MjModel.from_xml_path(scene_path)
data = mujoco.MjData(model)

# Launch Mujoco
with mujoco.viewer.launch(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()