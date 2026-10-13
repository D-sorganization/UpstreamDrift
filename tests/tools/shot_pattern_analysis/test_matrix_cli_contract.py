"""Matrix parsers preserve their public CLI contract without running studies."""

from pathlib import Path

import pytest

from src.tools.shot_pattern_analysis import matched_matrix, matched_supplement

pytestmark = pytest.mark.unit
MODULES = (matched_matrix, matched_supplement)


def _positionals(module):
    return ["pw_primary_face", "output"] if module is matched_supplement else ["output"]


@pytest.mark.parametrize("module", MODULES)
def test_matrix_parser_preserves_defaults_and_argument_order(module):
    parser = module.build_parser()
    parsed = vars(parser.parse_args(_positionals(module)))
    expected = {
        "output": Path("output"),
        "stage": "pilot",
        "shots": 32,
        "acceptance": None,
        "shard_index": 0,
        "shard_count": 1,
        "figures": "selected",
    }
    if module is matched_supplement:
        expected = {"protocol": "pw_primary_face", **expected}
    else:
        expected.update(
            common_targets=None,
            profiles=None,
            face_bounds_deg=None,
            speed_factors=(0.75, 1.25),
            cells_file=None,
            supplement_protocol=None,
        )
    assert parsed == expected
    assert [action.dest for action in parser._actions] == ["help", *expected]


@pytest.mark.parametrize("module", MODULES)
def test_matrix_parser_converts_shared_options_without_consuming_specific_ones(module):
    options = [
        "--stage",
        "final",
        "--shots",
        "10000",
        "--acceptance",
        "gate.json",
        "--shard-index",
        "2",
        "--shard-count",
        "4",
        "--figures",
        "none",
    ]
    args = module.build_parser().parse_args([*_positionals(module), *options])
    assert (args.stage, args.shots, args.acceptance) == (
        "final",
        10000,
        Path("gate.json"),
    )
    assert (args.shard_index, args.shard_count, args.figures) == (2, 4, "none")


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize(
    "option,value,message",
    [
        (
            "--stage",
            "unknown",
            "invalid choice: 'unknown'",
        ),
        ("--shots", "many", "invalid int value: 'many'"),
        (
            "--figures",
            "unknown",
            "invalid choice: 'unknown'",
        ),
    ],
)
def test_matrix_parser_preserves_argparse_rejections(
    module, option, value, message, capsys
):
    with pytest.raises(SystemExit) as error:
        module.build_parser().parse_args([*_positionals(module), option, value])
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    # Python releases vary the quoting of choices in argparse diagnostics.
    if option != "--shots":
        choices = (
            ("pilot", "final") if option == "--stage" else ("none", "selected", "all")
        )
        assert module.build_parser()._option_string_actions[option].choices == choices


@pytest.mark.parametrize("module", MODULES)
def test_matrix_main_dispatches_defaults_without_simulation(module, monkeypatch):
    calls = []
    name = (
        "run_supplement_matrix"
        if module is matched_supplement
        else "run_matched_matrix"
    )
    monkeypatch.setattr(
        module, name, lambda *args, **kwargs: calls.append((args, kwargs))
    )
    monkeypatch.setattr("sys.argv", ["study", *_positionals(module)])
    module.main()
    positional, options = calls[0]
    assert positional == (
        ("pw_primary_face", Path("output"))
        if module is matched_supplement
        else (Path("output"),)
    )
    assert {
        key: options[key]
        for key in (
            "n_shots",
            "stage",
            "acceptance",
            "shard_index",
            "shard_count",
            "figures",
        )
    } == {
        "n_shots": 32,
        "stage": "pilot",
        "acceptance": None,
        "shard_index": 0,
        "shard_count": 1,
        "figures": "selected",
    }
