"""Regression tests for Drake model handler."""

from src.launchers.launcher_model_handlers import ModelHandlerRegistry, ModuleHandler


class TestDrakeHandler:
    """Verify Drake uses ModuleHandler for correct relative-import support."""

    def test_drake_type_handled(self) -> None:
        """The registry must resolve a handler for 'drake'."""
        registry = ModelHandlerRegistry()
        handler = registry.get_handler("drake")
        assert handler is not None

    def test_drake_golf_type_handled(self) -> None:
        """The registry must resolve a handler for 'drake_golf'."""
        registry = ModelHandlerRegistry()
        handler = registry.get_handler("drake_golf")
        assert handler is not None

    def test_drake_is_module_handler(self) -> None:
        """Drake must use ModuleHandler (not ScriptHandler) for relative imports."""
        registry = ModelHandlerRegistry()
        handler = registry.get_handler("drake")
        assert isinstance(handler, ModuleHandler)
