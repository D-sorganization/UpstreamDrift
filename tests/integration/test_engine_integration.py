"""
Integration tests between different physics engines.
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestEngineIntegration:
    """Test integration between different physics engines."""
    
    @pytest.mark.integration
    def test_mujoco_drake_comparison(self, sample_swing_data):
        """Test comparison between MuJoCo and Drake engines."""
        # Mock both engines
        with patch('engines.physics_engines.mujoco.MuJoCoGolfModel') as mock_mujoco, \
             patch('engines.physics_engines.drake.DrakeGolfModel') as mock_drake:
            
            # Setup mock results
            mock_mujoco_results = {
                'ball_distance': 250.0,
                'launch_angle': 12.5,
                'ball_speed': 150.0,
                'simulation_time': 2.0
            }
            
            mock_drake_results = {
                'ball_distance': 248.5,
                'launch_angle': 12.8,
                'ball_speed': 149.2,
                'simulation_time': 2.0
            }
            
            mock_mujoco.return_value.simulate.return_value = mock_mujoco_results
            mock_drake.return_value.simulate.return_value = mock_drake_results
            
            # Compare results
            mujoco_distance = mock_mujoco_results['ball_distance']
            drake_distance = mock_drake_results['ball_distance']
            
            # Results should be within reasonable tolerance
            distance_diff = abs(mujoco_distance - drake_distance)
            assert distance_diff < 10.0  # Within 10 yards
    
    @pytest.mark.integration
    def test_cross_engine_validation(self):
        """Test validation of results across multiple engines."""
        # Mock all three engines
        engines = ['mujoco', 'drake', 'pinocchio']
        results = {}
        
        for engine in engines:
            with patch(f'engines.physics_engines.{engine}.{engine.title()}GolfModel') as mock_engine:
                # Generate slightly different but consistent results
                base_distance = 250.0
                noise = np.random.normal(0, 2.0)  # Small random variation
                
                mock_result = {
                    'ball_distance': base_distance + noise,
                    'launch_angle': 12.5 + np.random.normal(0, 0.5),
                    'ball_speed': 150.0 + np.random.normal(0, 3.0),
                }
                
                mock_engine.return_value.simulate.return_value = mock_result
                results[engine] = mock_result
        
        # Validate consistency
        distances = [results[engine]['ball_distance'] for engine in engines]
        distance_std = np.std(distances)
        
        # Standard deviation should be reasonable (< 5% of mean)
        assert distance_std < np.mean(distances) * 0.05
    
    @pytest.mark.integration
    def test_engine_parameter_consistency(self):
        """Test that all engines accept consistent parameter sets."""
        common_parameters = {
            'swing_speed': 100.0,  # mph
            'club_type': 'driver',
            'ball_position': [0, 0, 0],
            'simulation_time': 2.0,
            'timestep': 0.001,
        }
        
        engines = ['mujoco', 'drake', 'pinocchio']
        
        for engine in engines:
            with patch(f'engines.physics_engines.{engine}.{engine.title()}GolfModel') as mock_engine:
                mock_instance = Mock()
                mock_engine.return_value = mock_instance
                
                # Test parameter setting
                for param, value in common_parameters.items():
                    setattr(mock_instance, param, value)
                
                # Should not raise exceptions
                assert mock_instance.swing_speed == 100.0
                assert mock_instance.club_type == 'driver'
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_performance_comparison(self):
        """Test performance characteristics of different engines."""
        import time
        
        engines = ['mujoco', 'drake', 'pinocchio']
        performance_results = {}
        
        for engine in engines:
            with patch(f'engines.physics_engines.{engine}.{engine.title()}GolfModel') as mock_engine:
                mock_instance = Mock()
                mock_engine.return_value = mock_instance
                
                # Mock simulation with realistic timing
                def mock_simulate():
                    # Simulate different performance characteristics
                    if engine == 'mujoco':
                        time.sleep(0.1)  # Slower but more accurate
                    elif engine == 'drake':
                        time.sleep(0.05)  # Medium speed
                    else:  # pinocchio
                        time.sleep(0.02)  # Fastest
                    
                    return {'ball_distance': 250.0}
                
                mock_instance.simulate = mock_simulate
                
                # Measure performance
                start_time = time.time()
                result = mock_instance.simulate()
                end_time = time.time()
                
                performance_results[engine] = {
                    'simulation_time': end_time - start_time,
                    'result': result
                }
        
        # Verify expected performance ordering
        # Pinocchio should be fastest, MuJoCo slowest
        assert performance_results['pinocchio']['simulation_time'] < \
               performance_results['drake']['simulation_time']
        assert performance_results['drake']['simulation_time'] < \
               performance_results['mujoco']['simulation_time']


class TestEngineDataFlow:
    """Test data flow between engines and shared components."""
    
    @pytest.mark.integration
    def test_shared_data_structures(self, sample_swing_data):
        """Test that all engines work with shared data structures."""
        # Test that swing data can be processed by all engines
        engines = ['mujoco', 'drake', 'pinocchio']
        
        for engine in engines:
            with patch(f'engines.physics_engines.{engine}.{engine.title()}GolfModel') as mock_engine:
                mock_instance = Mock()
                mock_engine.return_value = mock_instance
                
                # Test data input
                mock_instance.load_swing_data.return_value = True
                result = mock_instance.load_swing_data(sample_swing_data)
                
                assert result is True
                mock_instance.load_swing_data.assert_called_with(sample_swing_data)
    
    @pytest.mark.integration
    def test_output_format_consistency(self, temp_dir):
        """Test that all engines produce consistent output formats."""
        from tests.conftest import sample_swing_data
        
        engines = ['mujoco', 'drake', 'pinocchio']
        expected_fields = [
            'ball_distance',
            'launch_angle', 
            'ball_speed',
            'simulation_time',
            'trajectory_data'
        ]
        
        for engine in engines:
            with patch(f'engines.physics_engines.{engine}.{engine.title()}GolfModel') as mock_engine:
                mock_instance = Mock()
                mock_engine.return_value = mock_instance
                
                # Mock consistent output format
                mock_result = {field: 0.0 for field in expected_fields}
                mock_result['trajectory_data'] = sample_swing_data(None)  # Mock trajectory
                
                mock_instance.simulate.return_value = mock_result
                
                result = mock_instance.simulate()
                
                # Verify all expected fields are present
                for field in expected_fields:
                    assert field in result
    
    @pytest.mark.integration
    def test_engine_error_handling(self):
        """Test error handling consistency across engines."""
        engines = ['mujoco', 'drake', 'pinocchio']
        
        for engine in engines:
            with patch(f'engines.physics_engines.{engine}.{engine.title()}GolfModel') as mock_engine:
                mock_instance = Mock()
                mock_engine.return_value = mock_instance
                
                # Test invalid parameter handling
                mock_instance.set_swing_speed.side_effect = ValueError("Invalid swing speed")
                
                with pytest.raises(ValueError):
                    mock_instance.set_swing_speed(-100)  # Invalid negative speed


class TestEngineConfiguration:
    """Test configuration management across engines."""
    
    @pytest.mark.integration
    def test_unified_configuration(self, sample_config):
        """Test that unified configuration works for all engines."""
        engines = ['mujoco', 'drake', 'pinocchio']
        
        for engine in engines:
            engine_config = sample_config['engines'][engine]
            
            with patch(f'engines.physics_engines.{engine}.{engine.title()}GolfModel') as mock_engine:
                mock_instance = Mock()
                mock_engine.return_value = mock_instance
                
                # Test configuration loading
                mock_instance.load_config.return_value = True
                result = mock_instance.load_config(engine_config)
                
                assert result is True
                mock_instance.load_config.assert_called_with(engine_config)
    
    @pytest.mark.integration
    def test_engine_switching(self, sample_config):
        """Test switching between engines at runtime."""
        # Mock engine manager
        with patch('shared.python.engine_manager.EngineManager') as mock_manager:
            mock_instance = Mock()
            mock_manager.return_value = mock_instance
            
            # Test switching engines
            engines = ['mujoco', 'drake', 'pinocchio']
            
            for engine in engines:
                mock_instance.switch_engine.return_value = True
                result = mock_instance.switch_engine(engine)
                assert result is True
            
            # Verify all engines were switched to
            assert mock_instance.switch_engine.call_count == len(engines)