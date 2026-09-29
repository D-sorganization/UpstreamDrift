"""Tests for model_generation.library._model_search."""

from __future__ import annotations

from model_generation.library._model_search import (
    get_categories,
    get_model,
    get_tags,
    iter_entries,
    list_models,
)
from model_generation.library._model_types import (
    ModelCategory,
    ModelEntry,
    RepositorySource,
)


def _make(
    id_: str,
    name: str,
    *,
    cat: ModelCategory = ModelCategory.OTHER,
    src: RepositorySource = RepositorySource.LOCAL,
    tags: list[str] | None = None,
    desc: str = "",
) -> ModelEntry:
    return ModelEntry(
        id=id_,
        name=name,
        description=desc,
        category=cat,
        source=src,
        tags=list(tags or []),
    )


def _sample() -> dict[str, ModelEntry]:
    return {
        "a": _make("a", "Alpha", cat=ModelCategory.HUMANOID, tags=["bio", "test"]),
        "b": _make(
            "b",
            "Bravo",
            cat=ModelCategory.ROBOT_ARM,
            src=RepositorySource.GITHUB,
            tags=["arm"],
            desc="A robotic arm",
        ),
        "c": _make("c", "Charlie", cat=ModelCategory.HUMANOID, tags=["bio"]),
    }


class TestListModels:
    def test_no_filter_returns_sorted_by_name(self) -> None:
        result = list_models(_sample())
        assert [e.name for e in result] == ["Alpha", "Bravo", "Charlie"]

    def test_filter_by_category(self) -> None:
        result = list_models(_sample(), category=ModelCategory.HUMANOID)
        assert {e.id for e in result} == {"a", "c"}

    def test_filter_by_source(self) -> None:
        result = list_models(_sample(), source=RepositorySource.GITHUB)
        assert [e.id for e in result] == ["b"]

    def test_filter_by_tags_any_match(self) -> None:
        result = list_models(_sample(), tags=["arm", "test"])
        assert {e.id for e in result} == {"a", "b"}

    def test_filter_by_tags_no_match(self) -> None:
        assert list_models(_sample(), tags=["nope"]) == []

    def test_search_in_name_case_insensitive(self) -> None:
        result = list_models(_sample(), search="ALPHA")
        assert [e.id for e in result] == ["a"]

    def test_search_in_description(self) -> None:
        result = list_models(_sample(), search="robotic")
        assert [e.id for e in result] == ["b"]

    def test_search_no_match(self) -> None:
        assert list_models(_sample(), search="zzzz") == []

    def test_combined_filters(self) -> None:
        result = list_models(
            _sample(),
            category=ModelCategory.HUMANOID,
            tags=["bio"],
            search="charlie",
        )
        assert [e.id for e in result] == ["c"]


class TestGetModel:
    def test_present(self) -> None:
        s = _sample()
        assert get_model(s, "b").id == "b"

    def test_missing(self) -> None:
        assert get_model(_sample(), "missing") is None


class TestGetCategoriesAndTags:
    def test_get_categories_sorted_unique(self) -> None:
        cats = get_categories(_sample())
        assert cats == sorted(cats, key=lambda c: c.value)
        assert ModelCategory.HUMANOID in cats
        assert ModelCategory.ROBOT_ARM in cats

    def test_get_tags_sorted_unique(self) -> None:
        assert get_tags(_sample()) == ["arm", "bio", "test"]

    def test_empty(self) -> None:
        assert get_categories({}) == []
        assert get_tags({}) == []


class TestIterEntries:
    def test_iterates_all(self) -> None:
        s = _sample()
        ids = {e.id for e in iter_entries(s)}
        assert ids == set(s.keys())
