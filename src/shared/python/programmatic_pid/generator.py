"""P&ID generator — backward-compatible shim module.

This module preserves the original public API by re-exporting from the
decomposed submodules.  New code should import from the specific modules
(``rendering``, ``layout``, ``streams``, etc.) or use the ``PIDDocument``
facade class.

.. deprecated:: 0.3.0
    Import from specific submodules instead of ``generator``.
"""

from __future__ import annotations

# CLI / orchestration
from src.shared.python.programmatic_pid.cli import (  # noqa: F401
    add_equipment,
    derive_related_path,
    generate,
    generate_controls_sheet,
    generate_process_sheet,
    main,
)

# Controls
from src.shared.python.programmatic_pid.controls import (  # noqa: F401
    add_control_loops,
    orthogonal_control_route,
    resolve_reference_point,
)

# Equipment
from src.shared.python.programmatic_pid.equipment import (  # noqa: F401
    EQUIPMENT_RENDERERS,
    draw_equipment_symbol,
    equipment_anchor,
    equipment_center,
    equipment_dims,
    equipment_side_anchors,
    nearest_equipment_anchor,
    register_equipment,
)
from src.shared.python.programmatic_pid.geometry import (  # noqa: F401
    clamp,
    closest_point_on_rect,
    dedupe_points,
    rects_overlap,
    text_box,
    to_float,
)

# Instruments
from src.shared.python.programmatic_pid.instruments import add_instrument  # noqa: F401

# Layout
from src.shared.python.programmatic_pid.layout import (  # noqa: F401
    LabelPlacer,
    compute_layout_regions,
    get_equipment_bounds,
    get_modelspace_extent,
    spread_instrument_positions,
)

# Profiles
from src.shared.python.programmatic_pid.profiles import (  # noqa: F401
    PROFILE_PRESETS,
    apply_profile,
)

# Rendering primitives
from src.shared.python.programmatic_pid.rendering import (  # noqa: F401
    add_arrow,
    add_arrow_head,
    add_box,
    add_poly_arrow,
    add_text,
    add_text_panel,
    ensure_layer,
    ensure_layers,
    export_svg_from_dxf,
    layer_name,
    parse_alignment,
    wrap_text_lines,
)

# Spec loading
from src.shared.python.programmatic_pid.spec_loader import (  # noqa: F401
    ensure_drawing,
    get_drawing,
    get_layer_config,
    get_layout_config,
    get_project,
    get_text_config,
    load_spec,
    prepare_spec,
)

# Streams
from src.shared.python.programmatic_pid.streams import (  # noqa: F401
    add_stream,
    resolve_endpoint,
)

# Title block & notes
from src.shared.python.programmatic_pid.title_block import (  # noqa: F401
    add_notes,
    add_title_block,
    get_mass_balance_values,
)

# Re-export everything for backward compatibility --------------------------
# Types & geometry
from src.shared.python.programmatic_pid.types import SpecValidationError  # noqa: F401

# Validation
from src.shared.python.programmatic_pid.validation import validate_spec  # noqa: F401
