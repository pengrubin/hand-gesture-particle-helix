#!/usr/bin/env python3
"""
Quick Integration Test
Tests the core integration functionality without camera/GUI dependencies.
"""

import numpy as np
import sys
import os
sys.path.append('/Users/hongweipeng/hand-gesture-particle-helix')

from gesture_radius_mapper import GestureToRadiusMapper, HandState


def test_twelve_tone_mapping():
    """Test twelve-tone scale mapping accuracy."""
    print("Testing twelve-tone scale mapping...")
    
    mapper = GestureToRadiusMapper(r_max=2.0)
    
    # Test finger count to radius mapping
    print("\nTwelve-Tone Scale Mapping:")
    print("Fingers | Radius | Frequency | Ratio | Semitones")
    print("-" * 50)
    
    all_tests_pass = True
    
    for finger_count in range(6):  # 0-5 fingers
        radius = mapper.finger_count_to_radius(finger_count)
        frequency = mapper.finger_count_to_frequency(finger_count)
        
        # Calculate expected values using twelve-tone formula
        exponent = (finger_count - 5) * 2 / 12  # 5 is reference
        expected_radius = mapper.r_max * (2 ** exponent)
        ratio = radius / mapper.r_max
        semitone_offset = (finger_count - 5) * 2
        
        print(f"   {finger_count}    | {radius:.3f} |  {frequency:.3f}   | {ratio:.3f} |    {semitone_offset:+d}")
        
        # Verify twelve-tone relationship
        if abs(radius - expected_radius) > 0.001:
            print(f"    ERROR: Expected {expected_radius:.3f}, got {radius:.3f}")
            all_tests_pass = False
            
        # Verify reasonable ranges
        if radius <= 0 or frequency <= 0:
            print(f"    ERROR: Invalid values - radius: {radius}, frequency: {frequency}")
            all_tests_pass = False
    
    return all_tests_pass


def test_hand_assignments():
    """Test different hand assignment strategies."""
    print("\nTesting hand assignment strategies...")
    
    mapper = GestureToRadiusMapper(r_max=2.0)
    
    # Test scenarios
    test_scenarios = [
        ("Both hands active", "three_fingers", "two_fingers", (0.2, 0.3), (-0.1, 0.4)),
        ("Left hand only", "open_hand", None, (0.1, 0.2), None),
        ("Right hand only", None, "one_finger", None, (-0.2, 0.1)),
        ("No hands", None, None, None, None),
    ]
    
    all_tests_pass = True
    
    for scenario_name, left_gesture, right_gesture, left_pos, right_pos in test_scenarios:
        print(f"\n  Scenario: {scenario_name}")
        
        # Convert None to proper gesture strings
        left_hand_gesture = left_gesture if left_gesture else "no_hand"
        right_hand_gesture = right_gesture if right_gesture else "no_hand"
        left_hand_position = left_pos if left_pos else (0.0, 0.0)
        right_hand_position = right_pos if right_pos else (0.0, 0.0)
        
        mapper.update_hand_states(
            left_hand_gesture=left_hand_gesture,
            right_hand_gesture=right_hand_gesture,
            left_hand_position=left_hand_position,
            right_hand_position=right_hand_position
        )
        
        params = mapper.get_parameters()
        
        print(f"    r1={params['r1']:.3f}, r2={params['r2']:.3f}, w1={params['w1']:.3f}, w2={params['w2']:.3f}")
        
        # Verify parameter ranges
        if not (0 < params['r1'] < 10 and 0 < params['r2'] < 10):
            print(f"    ERROR: Invalid radius values")
            all_tests_pass = False
            
        if not (0 < params['w1'] < 50 and 0 < params['w2'] < 50):
            print(f"    ERROR: Invalid frequency values")
            all_tests_pass = False
    
    return all_tests_pass


def test_parameter_smoothing():
    """Test parameter smoothing functionality."""
    print("\nTesting parameter smoothing...")
    
    mapper = GestureToRadiusMapper(r_max=2.0)
    mapper.set_smoothing_factor(0.8)  # High smoothing
    
    # Start with no hands
    mapper.update_hand_states("no_hand", "no_hand")
    initial_params = mapper.get_parameters()
    
    # Sudden change to maximum gesture
    mapper.update_hand_states("open_hand", "open_hand") 
    
    # Track parameter evolution
    param_history = []
    for i in range(10):
        params = mapper.get_parameters()
        param_history.append(params['r1'])
        
    print(f"  Parameter evolution: {param_history[:5]}...")
    
    # Check that change is gradual
    if len(param_history) >= 3:
        # First change should be smaller than total change
        first_change = abs(param_history[1] - param_history[0])
        total_change = abs(param_history[-1] - param_history[0])
        
        gradual_change = first_change < total_change * 0.5  # First step < 50% of total
        print(f"  Gradual change: {gradual_change} (first: {first_change:.3f}, total: {total_change:.3f})")
        return gradual_change
    
    return False


def test_gesture_state_mapping():
    """Test gesture string to HandState mapping."""
    print("\nTesting gesture state mapping...")
    
    mapper = GestureToRadiusMapper()
    
    # Test gesture mappings
    gesture_tests = [
        ("no_hand", HandState.NO_HAND, 0),
        ("fist", HandState.FIST, 0),
        ("one_finger", HandState.ONE_FINGER, 1),
        ("two_fingers", HandState.TWO_FINGERS, 2),
        ("three_fingers", HandState.THREE_FINGERS, 3),
        ("four_fingers", HandState.FOUR_FINGERS, 4),
        ("open_hand", HandState.OPEN_HAND, 5),
    ]
    
    all_tests_pass = True
    
    for gesture_str, expected_state, expected_fingers in gesture_tests:
        # Check HandState mapping
        finger_count = mapper.hand_state_to_finger_count.get(expected_state, -1)
        
        print(f"  {gesture_str:12} -> {expected_state.value:12} -> {finger_count} fingers")
        
        if finger_count != expected_fingers:
            print(f"    ERROR: Expected {expected_fingers} fingers, got {finger_count}")
            all_tests_pass = False
    
    return all_tests_pass


def main():
    """Run quick integration tests."""
    print("=" * 60)
    print("QUICK INTEGRATION TEST")
    print("=" * 60)
    
    tests = [
        ("Twelve-tone mapping", test_twelve_tone_mapping),
        ("Hand assignments", test_hand_assignments), 
        ("Parameter smoothing", test_parameter_smoothing),
        ("Gesture state mapping", test_gesture_state_mapping),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            status = "PASS" if result else "FAIL"
            print(f"\n{test_name}: {status}")
        except Exception as e:
            print(f"\n{test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20}: {status}")
    
    print("=" * 60)
    overall_status = "PASS" if passed == total else "FAIL"
    print(f"OVERALL: {passed}/{total} [{overall_status}]")
    print("=" * 60)
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())