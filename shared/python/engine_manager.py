"""
Engine Manager for Golf Modeling Suite.

Provides unified interface for managing and switching between different physics engines.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum

from .common_utils import setup_logging, GolfModelingError

logger = setup_logging(__name__)


class EngineType(Enum):
    """Available physics engines."""
    MUJOCO = "mujoco"
    DRAKE = "drake"
    PINOCCHIO = "pinocchio"
    MATLAB_2D = "matlab_2d"
    MATLAB_3D = "matlab_3d"
    PENDULUM = "pendulum"


class EngineStatus(Enum):
    """Engine status states."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


class EngineManager:
    """Manages physics engines for the Golf Modeling Suite."""
    
    def __init__(self, suite_root: Optional[Path] = None):
        """Initialize the engine manager.
        
        Args:
            suite_root: Root directory of the Golf Modeling Suite
        """
        self.suite_root = suite_root or Path(__file__).parent.parent.parent
        self.engines_root = self.suite_root / "engines"
        self.current_engine: Optional[EngineType] = None
        self.engine_instances: Dict[EngineType, Any] = {}
        self.engine_status: Dict[EngineType, EngineStatus] = {}
        
        # Initialize engine status
        self._check_engine_availability()
        
    def _check_engine_availability(self) -> None:
        """Check which engines are available."""
        engine_paths = {
            EngineType.MUJOCO: self.engines_root / "physics_engines" / "mujoco",
            EngineType.DRAKE: self.engines_root / "physics_engines" / "drake", 
            EngineType.PINOCCHIO: self.engines_root / "physics_engines" / "pinocchio",
            EngineType.MATLAB_2D: self.engines_root / "Simscape_Multibody_Models" / "2D_Golf_Model",
            EngineType.MATLAB_3D: self.engines_root / "Simscape_Multibody_Models" / "3D_Golf_Model",
            EngineType.PENDULUM: self.engines_root / "pendulum_models",
        }
        
        for engine_type, engine_path in engine_paths.items():
            if engine_path.exists():
                self.engine_status[engine_type] = EngineStatus.AVAILABLE
                logger.info(f"Engine {engine_type.value} is available at {engine_path}")
            else:
                self.engine_status[engine_type] = EngineStatus.UNAVAILABLE
                logger.warning(f"Engine {engine_type.value} not found at {engine_path}")
    
    def get_available_engines(self) -> List[EngineType]:
        """Get list of available engines.
        
        Returns:
            List of available engine types
        """
        return [
            engine_type for engine_type, status in self.engine_status.items()
            if status == EngineStatus.AVAILABLE
        ]
    
    def switch_engine(self, engine_type: EngineType) -> bool:
        """Switch to a different physics engine.
        
        Args:
            engine_type: The engine to switch to
            
        Returns:
            True if switch was successful, False otherwise
        """
        if engine_type not in self.engine_status:
            logger.error(f"Unknown engine type: {engine_type}")
            return False
            
        if self.engine_status[engine_type] != EngineStatus.AVAILABLE:
            logger.error(f"Engine {engine_type.value} is not available")
            return False
        
        try:
            # Set status to loading
            self.engine_status[engine_type] = EngineStatus.LOADING
            
            # Load engine if not already loaded
            if engine_type not in self.engine_instances:
                self._load_engine(engine_type)
            
            # Switch to engine
            self.current_engine = engine_type
            self.engine_status[engine_type] = EngineStatus.READY
            
            logger.info(f"Successfully switched to engine: {engine_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch to engine {engine_type.value}: {e}")
            self.engine_status[engine_type] = EngineStatus.ERROR
            return False
    
    def _load_engine(self, engine_type: EngineType) -> None:
        """Load a specific engine.
        
        Args:
            engine_type: The engine to load
            
        Raises:
            GolfModelingError: If engine loading fails
        """
        try:
            if engine_type == EngineType.MUJOCO:
                # Import MuJoCo engine components
                from engines.physics_engines.mujoco.python.mujoco_golf_pendulum import advanced_gui
                self.engine_instances[engine_type] = advanced_gui
                
            elif engine_type == EngineType.DRAKE:
                # Import Drake engine components
                from engines.physics_engines.drake.python.src import golf_gui
                self.engine_instances[engine_type] = golf_gui
                
            elif engine_type == EngineType.PINOCCHIO:
                # Import Pinocchio engine components
                from engines.physics_engines.pinocchio.python import main
                self.engine_instances[engine_type] = main
                
            elif engine_type in [EngineType.MATLAB_2D, EngineType.MATLAB_3D]:
                # MATLAB engines are handled differently (via MATLAB Engine API)
                self.engine_instances[engine_type] = "matlab_engine_placeholder"
                
            elif engine_type == EngineType.PENDULUM:
                # Import pendulum models
                from engines.pendulum_models.python import pendulum_main
                self.engine_instances[engine_type] = pendulum_main
                
            else:
                raise GolfModelingError(f"Unknown engine type: {engine_type}")
                
        except ImportError as e:
            raise GolfModelingError(f"Failed to import engine {engine_type.value}: {e}")
        except Exception as e:
            raise GolfModelingError(f"Failed to load engine {engine_type.value}: {e}")
    
    def get_current_engine(self) -> Optional[EngineType]:
        """Get the currently active engine.
        
        Returns:
            Current engine type or None if no engine is active
        """
        return self.current_engine
    
    def get_engine_status(self, engine_type: EngineType) -> EngineStatus:
        """Get the status of a specific engine.
        
        Args:
            engine_type: The engine to check
            
        Returns:
            Engine status
        """
        return self.engine_status.get(engine_type, EngineStatus.UNAVAILABLE)
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about all engines.
        
        Returns:
            Dictionary with engine information
        """
        return {
            "current_engine": self.current_engine.value if self.current_engine else None,
            "available_engines": [e.value for e in self.get_available_engines()],
            "engine_status": {e.value: s.value for e, s in self.engine_status.items()},
            "engines_root": str(self.engines_root),
        }
    
    def validate_engine_configuration(self, engine_type: EngineType) -> bool:
        """Validate that an engine is properly configured.
        
        Args:
            engine_type: The engine to validate
            
        Returns:
            True if engine is properly configured, False otherwise
        """
        if engine_type not in self.engine_status:
            return False
            
        if self.engine_status[engine_type] != EngineStatus.AVAILABLE:
            return False
        
        # Additional validation logic can be added here
        # For now, just check if the engine directory exists
        engine_paths = {
            EngineType.MUJOCO: self.engines_root / "physics_engines" / "mujoco" / "python",
            EngineType.DRAKE: self.engines_root / "physics_engines" / "drake" / "python",
            EngineType.PINOCCHIO: self.engines_root / "physics_engines" / "pinocchio" / "python",
            EngineType.MATLAB_2D: self.engines_root / "Simscape_Multibody_Models" / "2D_Golf_Model" / "matlab",
            EngineType.MATLAB_3D: self.engines_root / "Simscape_Multibody_Models" / "3D_Golf_Model" / "matlab",
            EngineType.PENDULUM: self.engines_root / "pendulum_models" / "python",
        }
        
        engine_path = engine_paths.get(engine_type)
        if engine_path and engine_path.exists():
            logger.info(f"Engine {engine_type.value} configuration is valid")
            return True
        else:
            logger.warning(f"Engine {engine_type.value} configuration is invalid")
            return False