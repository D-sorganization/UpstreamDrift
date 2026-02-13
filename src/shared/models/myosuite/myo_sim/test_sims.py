import logging
import os
import unittest

import mujoco

logger = logging.getLogger(__name__)

model_paths = [
    # Basic models
    "basic/myomuscle.xml",
    # finger models
    "finger/finger_v0.xml",
    "finger/myofinger_v0.xml",
    "finger/motorfinger_v0.xml",
    # elbow models
    "elbow/myoelbow_1dof6muscles_1dofexo.xml",
    "elbow/myoelbow_1dof6muscles.xml",
    "elbow/myoelbow_2dof6muscles.xml",
    "elbow/myoelbow_1dof6muscles_1dofSoftexo_Ideal.xml",
    "elbow/myoelbow_1dof6muscles_1dofSoftexo_sim2.xml",
    # arms
    "arm/myoarm_simple.xml",
    "arm/myoarm.xml",
    # hand models
    "hand/myohand.xml",
    # leg models
    "leg/myolegs.xml",
    "leg/myolegs_abdomen.xml",
    "osl/myolegs_osl.xml",
    # head
    "head/myohead_simple.xml",
    # torso
    "torso/myotorso.xml",
    "torso/myotorso_exosuit.xml",
    "torso/myotorso_rigid.xml",
    "torso/myotorso_abdomen.xml",
    # full body models
    "body/myobody.xml",
    "body/myoupperbody.xml",
    # scene
    "scene/myosuite_scene_noPedestal.xml",
    "scene/myosuite_scene.xml",
    "scene/myosuite_quad.xml",
    "scene/myosuite_logo.xml",
]


class TestSims(unittest.TestCase):
    def get_sim(self, model_path: str | None = None, model_xmlstr: str | None = None):
        """
        Get sim using model_path or model_xmlstr.
        """

        # load from path
        if model_path:
            # resolve full path
            if model_path.startswith("/"):
                fullpath = model_path
            else:
                fullpath = os.path.join(os.path.dirname(__file__), model_path)
            if not os.path.exists(fullpath):
                raise OSError(f"File {fullpath} does not exist")

            # load model
            if model_path.endswith(".mjb"):
                model = mujoco.MjModel.from_binary_path(fullpath)
            elif model_path.endswith(".xml"):
                model = mujoco.MjModel.from_xml_path(fullpath)

        # load from xml string
        elif model_xmlstr:
            model = mujoco.MjModel.from_xml_path(model_xmlstr)
        else:
            raise TypeError("Both model_path and model_xmlstr can't be None")

        return model

    def test_sims(self):
        """Load and verify each registered MuJoCo model."""
        for model_path in model_paths:
            logger.info(f"Testing: {model_path}")
            self.get_sim(model_path)


if __name__ == "__main__":
    unittest.main()
