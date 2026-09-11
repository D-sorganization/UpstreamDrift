from pathlib import Path


def test_inline_hand_includes_tag_ordering(tmp_path):
    """
    Test that inline_hand_includes correctly extracts <contact>, <tendon>,
    <actuator>, and <equality> blocks from inlined hand XMLs and appends
    them to the end of the scene XML to avoid MuJoCo parser buffer overruns.
    """

    # Mock _get_hand_content to return a mock hand XML with tags out of order
    def mock_get_hand_content(folder, filename, pattern, is_both):
        return """
<worldbody>
    <body name="rh_forearm">
        <geom name="forearm_geom"/>
    </body>
</worldbody>
<contact>
    <exclude body1="rh_wrist" body2="rh_forearm"/>
</contact>
<tendon>
    <fixed name="rh_FFJ0"/>
</tendon>
<actuator>
    <position name="rh_A_WRJ2"/>
</actuator>
<equality>
    <weld name="weld_test"/>
</equality>
"""

    # Inject a dummy implementation of inline_hand_includes from the actual source
    # We need to mock _get_hand_content which is used inside inline_hand_includes
    # but inline_hand_includes is a top-level function, so we should mock it on the module
    import src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf._grip_modelling_xml as mod
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf._grip_modelling_xml import (
        inline_hand_includes,
    )

    # Save original
    original_get = mod.get_hand_content
    mod.get_hand_content = mock_get_hand_content

    try:
        scene_xml = """<mujoco model="test">
        <include file="right_hand.xml"/>
        <worldbody>
            <geom name="scene_geom"/>
        </worldbody>
    </mujoco>"""

        result = inline_hand_includes(
            scene_xml, Path("dummy_scene.xml"), tmp_path, False
        )
    finally:
        # Restore
        mod.get_hand_content = original_get

    # Verify the include was removed
    assert '<include file="right_hand.xml"/>' not in result

    # Verify worldbody bodies were injected
    assert '<geom name="forearm_geom"/>' in result
    assert '<geom name="scene_geom"/>' in result

    # Verify that contact, tendon, actuator, equality are at the end, AFTER worldbody
    worldbody_end = result.find("</worldbody>")
    contact_start = result.find("<contact>")
    tendon_start = result.find("<tendon>")
    actuator_start = result.find("<actuator>")
    equality_start = result.find("<equality>")

    assert worldbody_end != -1
    assert contact_start > worldbody_end, (
        "<contact> should be placed after </worldbody>"
    )
    assert tendon_start > worldbody_end, "<tendon> should be placed after </worldbody>"
    assert actuator_start > worldbody_end, (
        "<actuator> should be placed after </worldbody>"
    )
    assert equality_start > worldbody_end, (
        "<equality> should be placed after </worldbody>"
    )
    assert result.endswith("</mujoco>"), "The file should end with </mujoco>"
