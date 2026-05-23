from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.collision_risk import CollisionRiskPredictor
from src.core.stgat import VehicleInteractionGraph


def test_collision_checkpoint_roundtrip(tmp_path: Path):
    ckpt = tmp_path / "collision.pt"

    predictor = CollisionRiskPredictor(device="cpu")
    predictor.save_model(str(ckpt), extra={"tag": "unit-test"})

    reloaded = CollisionRiskPredictor(device="cpu")
    assert reloaded.load_model(str(ckpt)) is True


def test_stgat_checkpoint_roundtrip(tmp_path: Path):
    ckpt = tmp_path / "stgat.pt"

    graph = VehicleInteractionGraph(device="cpu")
    graph.save_model(str(ckpt), extra={"tag": "unit-test"})

    reloaded = VehicleInteractionGraph(device="cpu")
    assert reloaded.load_model(str(ckpt)) is True
