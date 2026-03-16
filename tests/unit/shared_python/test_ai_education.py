from __future__ import annotations

from src.shared.python.ai.education import EducationSystem, GlossaryEntry
from src.shared.python.ai.types import ExpertiseLevel


def test_glossary_entry_get_definition():
    entry = GlossaryEntry(
        term="test",
        category="cat",
        definitions={
            ExpertiseLevel.BEGINNER: "beg",
            ExpertiseLevel.ADVANCED: "adv",
        },
        formula="f=ma",
        units="N",
    )

    assert entry.get_definition(ExpertiseLevel.BEGINNER) == "beg"
    assert entry.get_definition(ExpertiseLevel.INTERMEDIATE) == "beg"
    assert entry.get_definition(ExpertiseLevel.ADVANCED) == "adv"
    assert entry.get_definition(ExpertiseLevel.EXPERT) == "adv"


def test_education_system_init():
    edu = EducationSystem()
    assert len(edu) > 0
    assert "inverse_dynamics" in edu


def test_education_system_explain():
    edu = EducationSystem()

    # Not found
    assert edu.explain("not_found") == "Term 'not_found' not found in glossary."

    # Beginner
    exp = edu.explain("inverse dynamics", ExpertiseLevel.BEGINNER)
    assert "detective" in exp
    assert "Formula" not in exp
    assert "Units" not in exp

    # Advanced (shows formula and units if entry has it)
    exp_adv = edu.explain("inverse_dynamics", ExpertiseLevel.ADVANCED)
    assert "M(q)q\u0308" in exp_adv
    assert "Formula: " in exp_adv
    assert "Units: " in exp_adv

    # Inter (shows units but not formula typically, unless custom)
    exp_inter = edu.explain("inverse_dynamics", ExpertiseLevel.INTERMEDIATE)
    assert "Formula" not in exp_inter
    assert "Units" in exp_inter


def test_education_system_get_entry():
    edu = EducationSystem()
    entry = edu.get_entry("inverse dynamics")
    assert entry is not None
    assert entry.term == "Inverse Dynamics"

    assert edu.get_entry("missing") is None


def test_education_system_get_related_terms():
    edu = EducationSystem()
    rel = edu.get_related_terms("inverse dynamics")
    assert "forward_dynamics" in rel

    assert edu.get_related_terms("missing") == []


def test_education_system_search():
    edu = EducationSystem()
    results = edu.search("detective")  # in beginner inverse dynamics definition
    assert len(results) > 0
    assert results[0].term == "Inverse Dynamics"

    results = edu.search("pinocchio")
    assert any(r.term == "Pinocchio" for r in results)

    # Empty
    assert edu.search("xyznonexistent") == []


def test_education_system_categories():
    edu = EducationSystem()
    cats = edu.list_categories()
    assert "dynamics" in cats
    assert "kinematics" in cats
    assert "golf" in cats


def test_education_system_list_terms():
    edu = EducationSystem()
    all_t = edu.list_terms()
    assert "inverse_dynamics" in all_t

    cat_t = edu.list_terms("golf")
    assert "kinetic_chain" in cat_t
    assert "inverse_dynamics" not in cat_t


def test_education_system_add_entry():
    edu = EducationSystem()
    entry = GlossaryEntry(
        term="New Term", category="cat", definitions={ExpertiseLevel.BEGINNER: "new"}
    )
    edu.add_entry(entry)

    assert "new_term" in edu
    assert edu.get_entry("new term") == entry


def test_load_data_file_entries(monkeypatch):
    # Coverage for the try-except logic

    # We mock get_core_entries and get_extended_entries
    def mock_get_core():
        return [{"key": "core_mock", "term": "Core Mock", "cat": "test", "b": "beg"}]

    def mock_get_ext():
        return [{"key": "ext_mock", "term": "Ext Mock", "cat": "test", "i": "int"}]

    monkeypatch.setattr(
        "src.shared.python.ai.glossary_data_core.get_core_entries",
        mock_get_core,
        raising=False,
    )
    monkeypatch.setattr(
        "src.shared.python.ai.glossary_data_extended.get_extended_entries",
        mock_get_ext,
        raising=False,
    )

    edu = EducationSystem()
    assert "core_mock" in edu
    assert "ext_mock" in edu
