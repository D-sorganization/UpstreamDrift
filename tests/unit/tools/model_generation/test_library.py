"""
Tests for the model library module.
"""

import tempfile
from pathlib import Path


class TestModelLibrary:
    """Tests for ModelLibrary class."""

    def test_library_creation(self) -> None:
        """Test library instantiation."""
        from model_generation.library import ModelLibrary

        library = ModelLibrary()
        assert library is not None

    def test_list_models_empty(self) -> None:
        """Test listing models on fresh library."""
        from model_generation.library import ModelLibrary

        library = ModelLibrary()
        models = library.list_models()
        # Should return empty or built-in models
        assert isinstance(models, list)

    def test_add_local_model(self) -> None:
        """Test adding a local URDF model."""
        from model_generation.library import ModelCategory, ModelLibrary

        library = ModelLibrary()

        # Create a simple URDF
        urdf_content = """<?xml version="1.0"?>
        <robot name="test_robot">
            <link name="base_link">
                <inertial>
                    <mass value="1.0"/>
                    <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
                </inertial>
            </link>
        </robot>
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            f.write(urdf_content)
            temp_path = Path(f.name)

        try:
            entry = library.add_local_model(
                urdf_path=temp_path,
                name="test_robot",
                category=ModelCategory.HUMANOID,
                tags=["test", "simple"],
            )

            assert entry is not None
            assert entry.name == "test_robot"
            assert entry.category == ModelCategory.HUMANOID
            assert "test" in entry.tags
        finally:
            temp_path.unlink()

    def test_list_models_with_filter(self) -> None:
        """Test filtering models by category."""
        from model_generation.library import ModelCategory, ModelLibrary

        library = ModelLibrary()
        models = library.list_models(category=ModelCategory.HUMANOID)
        # All returned should be humanoid category
        for m in models:
            if m.category:
                assert m.category == ModelCategory.HUMANOID


class TestModelCache:
    """Tests for ModelCache class."""

    def test_cache_creation(self) -> None:
        """Test cache instantiation."""
        from model_generation.library.cache import CacheConfig, ModelCache

        config = CacheConfig(
            cache_dir=Path(tempfile.mkdtemp()),
            max_size_mb=100,
        )
        cache = ModelCache(config)
        assert cache is not None

    def test_cache_put_get(self) -> None:
        """Test adding and retrieving from cache."""
        from model_generation.library.cache import CacheConfig, ModelCache

        cache_dir = Path(tempfile.mkdtemp())
        config = CacheConfig(cache_dir=cache_dir, max_size_mb=100)
        cache = ModelCache(config)

        # Create test file
        test_file = cache_dir / "test_model" / "robot.urdf"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("<robot name='test'></robot>")

        # Put in cache
        entry = cache.put(
            model_id="test_model",
            local_path=test_file,
            source_url="http://example.com/robot.urdf",
        )

        assert entry is not None
        assert entry.model_id == "test_model"

        # Retrieve
        retrieved = cache.get("test_model")
        assert retrieved is not None
        assert retrieved.model_id == "test_model"

    def test_cache_contains(self) -> None:
        """Test cache contains check."""
        from model_generation.library.cache import CacheConfig, ModelCache

        cache_dir = Path(tempfile.mkdtemp())
        config = CacheConfig(cache_dir=cache_dir)
        cache = ModelCache(config)

        # Create and cache a file
        test_file = cache_dir / "existing_model.urdf"
        test_file.write_text("<robot></robot>")
        cache.put("existing", test_file)

        assert cache.contains("existing")
        assert not cache.contains("nonexistent")

    def test_cache_statistics(self) -> None:
        """Test cache statistics."""
        from model_generation.library.cache import CacheConfig, ModelCache

        cache_dir = Path(tempfile.mkdtemp())
        config = CacheConfig(cache_dir=cache_dir)
        cache = ModelCache(config)

        stats = cache.get_statistics()
        assert "entry_count" in stats
        assert "total_size_bytes" in stats
        assert "cache_dir" in stats


class TestRepository:
    """Tests for Repository classes."""

    def test_local_repository(self) -> None:
        """Test LocalRepository."""
        from model_generation.library.repository import LocalRepository

        repo_dir = Path(tempfile.mkdtemp())

        # Create some URDF files
        (repo_dir / "robot1.urdf").write_text("<robot name='robot1'></robot>")
        (repo_dir / "subdir").mkdir()
        (repo_dir / "subdir" / "robot2.urdf").write_text(
            "<robot name='robot2'></robot>"
        )

        repo = LocalRepository(repo_dir, name="test_local")
        models = repo.list_models()

        assert len(models) >= 2
        names = [m.name for m in models]
        assert "robot1" in names
        assert "robot2" in names

    def test_github_repository_list(self) -> None:
        """Test GitHubRepository model listing (mocked)."""
        from model_generation.library.repository import GitHubRepository

        # This test would normally mock the HTTP calls
        # For now, just verify instantiation
        repo = GitHubRepository(
            name="test_repo",
            owner="test_owner",
            repo="test_repo",
        )
        assert repo.name == "test_repo"
