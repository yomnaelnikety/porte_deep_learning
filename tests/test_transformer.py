import numpy as np
import pytest
import torch
import json
from argparse import Namespace
from dataclasses import asdict
import pandas as pd

from eventflow.transformer import EventTransformer
from eventflow.generate import diagnostics, js_divergence, ngram_counts, sample_events
from eventflow.train_transformer import checked_config
from eventflow.train_deep import _build_model
from test_seq_data import make_events, CLASSES
from eventflow.seq_data import Encoder


@pytest.fixture
def model():
    torch.manual_seed(22)
    return EventTransformer(3, [2], 2, hidden=16, heads=2, layers=2, dropout=0).eval()


def test_future_events_and_covariates_cannot_change_past(model):
    event = torch.tensor([[1, 2, 3, 1, 2, 3]])
    context = torch.ones(1, 6, 1, dtype=torch.long)
    numeric = torch.randn(1, 6, 2)
    original = model(event, context, numeric)
    event[:, 3:] = 2
    context[:, 3:] = 2
    numeric[:, 3:] += 100
    changed = model(event, context, numeric)
    torch.testing.assert_close(original[:, :3], changed[:, :3], atol=1e-6, rtol=1e-6)


def test_padding_and_batch_companions_do_not_change_predictions(model):
    event = torch.tensor([[1, 2, 3]])
    context = torch.ones(1, 3, 1, dtype=torch.long)
    numeric = torch.randn(1, 3, 2)
    alone = model(event, context, numeric)
    padded_event = torch.tensor([[1, 2, 3, 0, 0], [2, 2, 1, 3, 2]])
    padded_context = torch.ones(2, 5, 1, dtype=torch.long)
    padded_numeric = torch.randn(2, 5, 2)
    padded_numeric[0, :3] = numeric[0]
    batched = model(padded_event, padded_context, padded_numeric)
    torch.testing.assert_close(alone[0], batched[0, :3], atol=1e-6, rtol=1e-6)


def test_many_to_many_gradients_are_finite(model):
    model.train()
    event = torch.tensor([[1, 2, 3, 1], [2, 3, 0, 0]])
    logits = model(event, torch.ones(2, 4, 1, dtype=torch.long), torch.randn(2, 4, 2))
    target = torch.tensor([[1, 2, 0, 1], [2, 0, -100, -100]])
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_optimized_evaluation_matches_training_path(model):
    event = torch.tensor([[1, 2, 3, 1], [2, 3, 0, 0]])
    context = torch.ones(2, 4, 1, dtype=torch.long)
    numeric = torch.randn(2, 4, 2)
    with torch.no_grad():
        fast = model(event, context, numeric)
        enabled = torch.backends.mha.get_fastpath_enabled()
        try:
            torch.backends.mha.set_fastpath_enabled(False)
            slow = model(event, context, numeric)
        finally:
            torch.backends.mha.set_fastpath_enabled(enabled)
    torch.testing.assert_close(fast, slow, atol=2e-6, rtol=2e-6)


def test_event_only_model_ignores_all_future_covariates():
    events = make_events([130, 140])
    encoder = Encoder.fit(events, ["M00"], CLASSES)
    cfg = checked_config({"hidden": 16, "heads": 2}, generator=True)
    model = _build_model(cfg, encoder, 0).eval()
    assert cfg.numeric_groups() == [] and len(model.context_embeddings) == 0
    event = torch.tensor([[1, 2, 3]])
    numeric = torch.empty(1, 3, 0)
    a = model(event, torch.zeros(1, 3, 6, dtype=torch.long), numeric)
    b = model(event, torch.full((1, 3, 6), 999, dtype=torch.long), numeric)
    torch.testing.assert_close(a, b)


def test_sampling_is_reproducible_and_appends_its_own_events():
    class Uniform(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = []
        def forward(self, event, context, numeric):
            self.seen.append(event.clone())
            assert context.shape[-1] == numeric.shape[-1] == 0
            return torch.zeros(*event.shape, 3)
    model = Uniform()
    a = sample_events(model, [1, 3], 5, 8, 1, 22, torch.device("cpu"))
    for step in range(1, 5):
        np.testing.assert_array_equal(model.seen[step][:, 2:].numpy(), a[:, :step]+1)
    b = sample_events(Uniform(), [1, 3], 5, 8, 1, 22, torch.device("cpu"))
    np.testing.assert_array_equal(a, b)
    assert len({tuple(row) for row in a}) > 1


def test_order_diagnostic_detects_equal_marginals_wrong_order():
    real = np.array([[0, 1, 0, 1, 0, 1]])
    wrong = np.array([[0, 0, 0, 1, 1, 1]])
    assert js_divergence(ngram_counts(real, 1), ngram_counts(wrong, 1)) == 0
    assert js_divergence(ngram_counts(real, 2), ngram_counts(wrong, 2)) > 0
    assert ngram_counts([[0, 1], [2, 0]], 2) == {(0, 1): 1, (2, 0): 1}


def test_diversity_uses_matched_sample_sizes():
    real = np.array([[0, 1, 2, 0], [1, 2, 0, 1]])
    generated = np.repeat(real[:, None, :], 3, axis=1)
    summary, frame = diagnostics(real, generated)
    assert summary["js_1gram"]["mean"] == 0
    assert summary["unique_2gram_ratio"]["mean"] == summary["real_diversity"]["unique_2gram_ratio"]
    assert np.isclose(summary["within_prefix_duplicate_rate"], 2/3)
    assert len(frame) == 3


def test_invalid_dimensions_and_sampling_rejected():
    with pytest.raises(ValueError):
        checked_config({"hidden": 17, "heads": 4})
    with pytest.raises(ValueError):
        sample_events(None, [1], 20, 2, 0, 1, torch.device("cpu"))
    with pytest.raises(ValueError):
        diagnostics(np.zeros((2, 20)), np.zeros((2, 1, 20)))


def test_checkpoint_to_generation_outputs_and_overlap_guard(tmp_path, monkeypatch):
    from eventflow import generate
    events = make_events([130, 140, 150])
    encoder = Encoder.fit(events, ["M00"], CLASSES)
    cfg = checked_config({"hidden": 16, "layers": 1, "heads": 2}, generator=True)
    model = _build_model(cfg, encoder, 0)
    saved = dict(config=asdict(cfg), state=model.state_dict(), encoder=encoder.to_dict(),
                 temperature=1.0, numeric=[], fit_ids=["M00"], smoke=True)
    checkpoint = tmp_path / "generator.pt"
    torch.save(saved, checkpoint)
    manifest = pd.DataFrame(dict(sequence_id=["M01", "M02"], prefix_len=[90, 100], horizon=[4, 4]))
    manifest.to_csv(tmp_path / "rollout_manifest.csv", index=False)
    protocol = pd.DataFrame(dict(sequence_id=["M00", "M01", "M02"], role=["fit", "validation", "validation"]))
    monkeypatch.setattr(generate, "load_protocol", lambda path: (CLASSES, protocol))
    monkeypatch.setattr(generate, "load_events", lambda path: events)
    args = Namespace(output=tmp_path / "output", samples=3, temperature=1.0,
                     checkpoint=checkpoint, results=tmp_path, limit=0, data=tmp_path,
                     device="cpu", seed=22)
    generate.run_generation(args)
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["n_generated"] == 6 and summary["horizon"] == 4 and summary["smoke"]
    arrays = np.load(args.output / "rollouts.npz")
    expected = events[events.sequence_id == "M01"].event_type.astype(str).iloc[90:94]
    assert arrays["real"][0].tolist() == [CLASSES.index(c) for c in expected]
    assert arrays["generated"].shape == (2, 3, 4)
    with pytest.raises(FileExistsError):
        generate.run_generation(args)
    args.output = tmp_path / "overlap"
    saved["fit_ids"] = ["M00", "M01"]
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="overlap"):
        generate.run_generation(args)


def test_comparison_aligns_both_keys_and_checks_probabilities(tmp_path):
    from eventflow.train_transformer import comparison
    folder = tmp_path / "runs" / "reference"
    folder.mkdir(parents=True)
    (tmp_path / "classes.json").write_text(json.dumps(["A", "B"]))
    ref = pd.DataFrame(dict(sequence_id=["M1", "M1", "M2", "M2"],
                            prefix_len=[90, 100, 90, 100], target=["A", "B", "A", "B"],
                            p__A=[.8, .3, .7, .2], p__B=[.2, .7, .3, .8]))
    ref.to_csv(folder / "validation_predictions.csv", index=False)
    other = tmp_path / "other.csv"
    ref.iloc[::-1].to_csv(other, index=False)
    args = Namespace(out=tmp_path, results=tmp_path, other=[other])
    comparison(args, "reference")
    result = pd.read_csv(tmp_path / "comparison_reference.csv")
    assert np.allclose(result.delta_vs_transformer, 0)
    ref["p__A"] = 2.0
    ref.to_csv(other, index=False)
    with pytest.raises(ValueError, match="normalized"):
        comparison(args, "reference")
