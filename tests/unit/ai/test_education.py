"""Unit tests for EducationSystem."""

from __future__ import annotations

from src.shared.python.ai.education import EducationSystem, GlossaryEntry
from src.shared.python.ai.types import ExpertiseLevel


class TestGlossaryEntry:
    """Tests for GlossaryEntry."""

    def test_get_definition_exact_level(self) -> None:
        """Test getting definition at exact expertise level."""
        entry = GlossaryEntry(
            term="Test Term",
            category="test",
            definitions={
                ExpertiseLevel.BEGINNER: "Simple explanation",
                ExpertiseLevel.ADVANCED: "Complex explanation",
            },
        )
        assert entry.get_definition(ExpertiseLevel.BEGINNER) == "Simple explanation"
        assert entry.get_definition(ExpertiseLevel.ADVANCED) == "Complex explanation"

    def test_get_definition_fallback_to_lower(self) -> None:
        """Test falling back to lower level when exact not available."""
        entry = GlossaryEntry(
            term="Test Term",
            category="test",
            definitions={
                ExpertiseLevel.BEGINNER: "Simple explanation",
            },
        )
        # Should fall back to beginner when intermediate not defined
        result = entry.get_definition(ExpertiseLevel.INTERMEDIATE)
        assert result == "Simple explanation"

    def test_get_definition_not_available(self) -> None:
        """Test message when no definition available."""
        entry = GlossaryEntry(
            term="Test Term",
            category="test",
            definitions={},
        )
        result = entry.get_definition(ExpertiseLevel.BEGINNER)
        assert "not available" in result


class TestEducationSystem:
    """Tests for EducationSystem."""

    def test_initialization(self) -> None:
        """Test education system initializes with glossary."""
        edu = EducationSystem()
        assert len(edu) > 0

    def test_explain_beginner(self) -> None:
        """Test explaining at beginner level."""
        edu = EducationSystem()
        explanation = edu.explain("inverse_dynamics", ExpertiseLevel.BEGINNER)
        assert "detective" in explanation.lower()  # Uses analogy

    def test_explain_advanced(self) -> None:
        """Test explaining at advanced level."""
        edu = EducationSystem()
        explanation = edu.explain("inverse_dynamics", ExpertiseLevel.ADVANCED)
        assert "τ" in explanation  # Includes formula
        assert "Formula:" in explanation

    def test_explain_term_not_found(self) -> None:
        """Test explaining non-existent term."""
        edu = EducationSystem()
        explanation = edu.explain("nonexistent_term")
        assert "not found" in explanation.lower()

    def test_explain_normalizes_term(self) -> None:
        """Test that term lookup normalizes variations."""
        edu = EducationSystem()
        # All of these should work
        assert edu.explain("inverse_dynamics") == edu.explain("inverse dynamics")
        assert edu.explain("C3D File") != "not found"
        assert edu.explain("c3d_file") != "not found"

    def test_get_entry(self) -> None:
        """Test getting full glossary entry."""
        edu = EducationSystem()
        entry = edu.get_entry("inverse_dynamics")
        assert entry is not None
        assert entry.term == "Inverse Dynamics"
        assert entry.units == "N·m"

    def test_get_related_terms(self) -> None:
        """Test getting related terms."""
        edu = EducationSystem()
        related = edu.get_related_terms("inverse_dynamics")
        assert "forward_dynamics" in related
        assert "joint_torque" in related

    def test_search_by_term(self) -> None:
        """Test searching glossary by term name."""
        edu = EducationSystem()
        results = edu.search("dynamics")
        term_names = [r.term for r in results]
        assert any("Dynamics" in name for name in term_names)

    def test_search_by_category(self) -> None:
        """Test searching glossary by category."""
        edu = EducationSystem()
        results = edu.search("golf")
        assert len(results) >= 2  # At least x-factor and kinetic chain

    def test_list_categories(self) -> None:
        """Test listing all categories."""
        edu = EducationSystem()
        categories = edu.list_categories()
        assert "dynamics" in categories
        assert "simulation" in categories
        assert "golf" in categories

    def test_list_terms(self) -> None:
        """Test listing all terms."""
        edu = EducationSystem()
        terms = edu.list_terms()
        assert "inverse_dynamics" in terms
        assert "mujoco" in terms

    def test_list_terms_by_category(self) -> None:
        """Test listing terms filtered by category."""
        edu = EducationSystem()
        simulation_terms = edu.list_terms(category="simulation")
        assert "mujoco" in simulation_terms
        assert "drake" in simulation_terms
        assert "inverse_dynamics" not in simulation_terms

    def test_add_entry(self) -> None:
        """Test adding custom glossary entry."""
        edu = EducationSystem()
        original_count = len(edu)

        edu.add_entry(
            GlossaryEntry(
                term="Custom Term",
                category="custom",
                definitions={
                    ExpertiseLevel.BEGINNER: "A custom definition",
                },
            )
        )

        assert len(edu) == original_count + 1
        assert "custom_term" in edu

    def test_contains(self) -> None:
        """Test checking if term exists."""
        edu = EducationSystem()
        assert "inverse_dynamics" in edu
        assert "nonexistent" not in edu


class TestDefaultGlossary:
    """Tests for default glossary content."""

    def test_has_core_dynamics_terms(self) -> None:
        """Test that core dynamics terms exist."""
        edu = EducationSystem()
        assert "inverse_dynamics" in edu
        assert "forward_dynamics" in edu
        assert "joint_torque" in edu

    def test_has_physics_engine_terms(self) -> None:
        """Test that physics engine terms exist."""
        edu = EducationSystem()
        assert "mujoco" in edu
        assert "drake" in edu
        assert "pinocchio" in edu

    def test_has_golf_terms(self) -> None:
        """Test that golf-specific terms exist."""
        edu = EducationSystem()
        assert "kinetic_chain" in edu
        assert "x_factor" in edu
        assert "ground_reaction_force" in edu

    def test_has_validation_terms(self) -> None:
        """Test that validation terms exist."""
        edu = EducationSystem()
        assert "cross_engine_validation" in edu
        assert "drift_control_decomposition" in edu

    def test_all_entries_have_beginner_definition(self) -> None:
        """Test that all entries have at least beginner definition."""
        edu = EducationSystem()
        for term in edu.list_terms():
            entry = edu.get_entry(term)
            assert entry is not None
            # Get definition at beginner level
            definition = entry.get_definition(ExpertiseLevel.BEGINNER)
            assert "not available" not in definition.lower(), f"{term} missing beginner"
