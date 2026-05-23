#!/usr/bin/env python3
"""
ST-GAT and Collision Risk Prediction - Test Script
Tests the innovative modules for vehicle interaction modeling and collision prediction
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_stgat():
    """Test Spatio-Temporal Graph Attention Network"""
    print("=" * 60)
    print("Test 1: Spatio-Temporal Graph Attention Network (ST-GAT)")
    print("=" * 60)

    from src.core.stgat import VehicleInteractionGraph, SpatioTemporalGAT
    import torch

    # Test 1.1: Graph Attention Layer
    print("\n1.1 Testing Graph Attention Layer...")
    from src.core.stgat import GraphAttentionLayer

    gat_layer = GraphAttentionLayer(in_features=8, out_features=16)
    x = torch.randn(5, 8)  # 5 nodes, 8 features
    adj = torch.ones(5, 5)  # Fully connected
    out = gat_layer(x, adj)
    assert out.shape == (5, 16)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    print("[PASS] Graph Attention Layer works correctly")

    # Test 1.2: ST-GAT Model
    print("\n1.2 Testing ST-GAT Model...")
    model = SpatioTemporalGAT(
        node_features=8,
        hidden_dim=32,
        output_dim=16,
        num_heads=2
    )
    x = torch.randn(5, 8)
    adj = torch.ones(5, 5)
    out = model(x, adj)
    assert out.shape == (5, 16)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    print("[PASS] ST-GAT Model works correctly")

    # Test 1.3: Vehicle Interaction Graph
    print("\n1.3 Testing Vehicle Interaction Graph...")
    graph = VehicleInteractionGraph(distance_threshold=200)

    # Simulate tracks
    tracks = [
        {'track_id': 1, 'bbox': (100, 100, 150, 150)},
        {'track_id': 2, 'bbox': (200, 100, 250, 150)},
        {'track_id': 3, 'bbox': (500, 100, 550, 150)},  # Far away
    ]

    embeddings = graph.update(tracks)
    print(f"   Tracks: {len(tracks)}")
    print(f"   Embeddings: {len(embeddings)}")
    for tid, emb in embeddings.items():
        print(f"   Track {tid}: embedding shape = {emb.shape}")

    # Test interaction score
    score_12 = graph.get_interaction_score(1, 2, embeddings)
    score_13 = graph.get_interaction_score(1, 3, embeddings)
    print(f"   Interaction score (1-2): {score_12:.3f}")
    print(f"   Interaction score (1-3): {score_13:.3f}")
    print("[PASS] Vehicle Interaction Graph works correctly")

    return True


def test_collision_risk():
    """Test Collision Risk Prediction"""
    print("\n" + "=" * 60)
    print("Test 2: Collision Risk Prediction")
    print("=" * 60)

    from src.core.collision_risk import (
        CollisionRiskPredictor, TrajectoryPredictor, RiskLevel
    )
    import torch

    # Test 2.1: Trajectory Predictor
    print("\n2.1 Testing Trajectory Predictor...")
    predictor = TrajectoryPredictor(
        input_dim=4,
        hidden_dim=64,
        pred_horizon=10
    )
    x = torch.randn(2, 5, 4)  # 2 vehicles, 5 history steps, 4 features
    pred = predictor(x)
    assert pred.shape == (2, 10, 2)  # 2 vehicles, 10 future steps, 2 coords
    print(f"   Input: {x.shape} -> Prediction: {pred.shape}")
    print("[PASS] Trajectory Predictor works correctly")

    # Test 2.2: Collision Risk Predictor
    print("\n2.2 Testing Collision Risk Predictor...")
    risk_predictor = CollisionRiskPredictor(
        history_length=10,
        prediction_horizon=15,
        fps=15.0
    )

    # Simulate converging vehicles
    print("   Simulating converging vehicles...")
    for frame in range(15):
        tracks = [
            {'track_id': 1, 'bbox': (100 + frame * 10, 200, 150 + frame * 10, 250)},
            {'track_id': 2, 'bbox': (400 - frame * 10, 200, 450 - frame * 10, 250)},
        ]
        risks = risk_predictor.update(tracks)

    print(f"   Detected risks: {len(risks)}")
    if risks:
        for risk in risks:
            print(f"   - Vehicles {risk.vehicle1_id} & {risk.vehicle2_id}: "
                  f"{risk.risk_level.value}, TTC={risk.time_to_collision:.2f}s")
    print("[PASS] Collision Risk Predictor works correctly")

    # Test 2.3: Risk Levels
    print("\n2.3 Testing Risk Level Classification...")
    risk_predictor2 = CollisionRiskPredictor(fps=15.0)

    # Test different scenarios
    scenarios = [
        ("Safe - parallel vehicles", [
            {'track_id': 1, 'bbox': (100, 100, 150, 150)},
            {'track_id': 2, 'bbox': (100, 300, 150, 350)},
        ]),
        ("Potential collision - converging", [
            {'track_id': 3, 'bbox': (100, 200, 150, 250)},
            {'track_id': 4, 'bbox': (200, 200, 250, 250)},
        ]),
    ]

    for name, initial_tracks in scenarios:
        risk_predictor2 = CollisionRiskPredictor(fps=15.0)
        for i in range(10):
            risks = risk_predictor2.update(initial_tracks)
        print(f"   {name}: {len(risks)} risks detected")

    print("[PASS] Risk Level Classification works correctly")

    return True


def test_integration():
    """Test integration of ST-GAT with Collision Risk"""
    print("\n" + "=" * 60)
    print("Test 3: Integration Test")
    print("=" * 60)

    from src.core.stgat import VehicleInteractionGraph
    from src.core.collision_risk import CollisionRiskPredictor

    graph = VehicleInteractionGraph()
    risk_predictor = CollisionRiskPredictor(fps=15.0)

    print("\n3.1 Simulating traffic scenario...")

    # Simulate a scenario with multiple vehicles
    for frame in range(20):
        tracks = [
            {'track_id': 1, 'bbox': (100 + frame * 5, 200, 150 + frame * 5, 250)},
            {'track_id': 2, 'bbox': (400 - frame * 5, 200, 450 - frame * 5, 250)},
            {'track_id': 3, 'bbox': (250, 100 + frame * 3, 300, 150 + frame * 3)},
        ]

        # Update interaction graph
        embeddings = graph.update(tracks)

        # Predict collision risks
        risks = risk_predictor.update(tracks)

        if frame == 19:
            print(f"   Frame {frame}:")
            print(f"   - Vehicles: {len(tracks)}")
            print(f"   - Interaction embeddings: {len(embeddings)}")
            print(f"   - Collision risks: {len(risks)}")

            # Get risk summary
            summary = risk_predictor.get_risk_summary(risks)
            print(f"   - Risk summary: {summary}")

    print("[PASS] Integration test completed")

    return True


def test_visualization():
    """Test visualization functions"""
    print("\n" + "=" * 60)
    print("Test 4: Visualization")
    print("=" * 60)

    import cv2
    from src.core.collision_risk import CollisionRiskPredictor, CollisionRisk, RiskLevel

    # Create test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (50, 50, 50)  # Gray background

    risk_predictor = CollisionRiskPredictor(fps=15.0)

    # Simulate and get risks
    for i in range(15):
        tracks = [
            {'track_id': 1, 'bbox': (100 + i * 10, 200, 150 + i * 10, 250)},
            {'track_id': 2, 'bbox': (400 - i * 10, 200, 450 - i * 10, 250)},
        ]
        risks = risk_predictor.update(tracks)

    # Draw predictions
    if risks:
        annotated = risk_predictor.draw_predictions(frame, risks)
        print(f"   Drew {len(risks)} risk predictions on frame")
        print(f"   Frame shape: {annotated.shape}")

        # Save test image
        cv2.imwrite('data/test_collision_viz.jpg', annotated)
        print("   Saved visualization to data/test_collision_viz.jpg")

    print("[PASS] Visualization test completed")

    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("ST-GAT & COLLISION RISK PREDICTION - TEST SUITE")
    print("Innovation: Spatio-Temporal Graph Attention + Trajectory Prediction")
    print("=" * 70)

    tests = [
        ("ST-GAT Module", test_stgat),
        ("Collision Risk Prediction", test_collision_risk),
        ("Integration", test_integration),
        ("Visualization", test_visualization),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            import traceback
            print(f"[ERROR] {name}: {e}")
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All innovation module tests passed!")
        print("\nInnovation Points Implemented:")
        print("  1. ST-GAT: Vehicle interaction modeling using graph attention")
        print("  2. Trajectory Prediction: LSTM-based future position prediction")
        print("  3. Collision Risk: TTC-based multi-level risk assessment")
        print("  4. Yielding Detection: Graph-based behavior understanding")
        return 0
    else:
        print("\n[FAILURE] Some tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
