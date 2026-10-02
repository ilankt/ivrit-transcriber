from engine.progress import StageProgress


def test_estimate_resets_between_steps_and_excludes_pauses():
    now = [0.0]
    progress = StageProgress(clock=lambda: now[0])
    assert "Estimating" in progress.update(1, 0, 100)[1]
    now[0] = 10
    assert progress.update(1, 25, 100) == (25, "Step 1/3 — 25% of step — About 30s left in this step")
    now[0] = 70
    progress.exclude_duration(60)
    assert "About 30s" in progress.update(1, 25, 100)[1]
    assert "Step complete" in progress.update(1, 100, 100)[1]
    assert progress.update(2, 0, 200) == (0, "Step 2/3 — 0% of step — Estimating time remaining…")
    now[0] = 90
    assert "About 1m 0s" in progress.update(2, 50, 200)[1]


def test_retry_restarts_estimate_and_empty_input_is_safe():
    now = [1.0]
    progress = StageProgress(clock=lambda: now[0])
    progress.update(1, 0, 100)
    now[0] = 21
    assert "About 20s" in progress.update(1, 50, 100)[1]
    assert "Estimating" in progress.update(1, 0, 100)[1]
    assert progress.update(2, 0, 0)[0] == 0
    assert progress.update(3, 120, 100)[0] == 100
