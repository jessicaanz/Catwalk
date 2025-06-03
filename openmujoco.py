import mujoco
import mujoco.viewer
import time

# Setup Parameters
scene_path = "C:/Users/jessi/Desktop/CS188/final_project/cat_model/cat_mujoco.xml"
model = mujoco.MjModel.from_xml_path(scene_path)
data = mujoco.MjData(model)
use_keyframe = False

# Launch Mujoco
with mujoco.viewer.launch(model, data) as viewer:

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()