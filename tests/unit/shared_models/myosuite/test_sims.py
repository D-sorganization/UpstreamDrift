from __future__ import annotations

import logging
import unittest


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


if __name__ == "__main__":
    unittest.main()
