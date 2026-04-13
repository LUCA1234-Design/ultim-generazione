from pathlib import Path

from rl.ppo_agent import PPOAgent


def test_ppo_save_and_load_pretrained(tmp_path):
    model = PPOAgent()
    out = tmp_path / "ppo_pretrained.pt"
    assert model.save(str(out)) is True
    assert out.exists()

    loaded = PPOAgent()
    assert loaded.load_pretrained(str(out)) is True
