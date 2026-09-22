import numpy as np
import pandas as pd

from jevquant.risk import RiskConfig, RiskManager


def test_model_veto_can_only_reduce_exposure():
    rng = np.random.default_rng(1)
    syms = list("ABCDE")
    for trial in range(500):
        target = pd.Series(rng.normal(0, 1, 5), index=syms)
        adversarial = pd.Series(rng.normal(0, 3, 5), index=syms)  # garbage, even negative or > 1
        cfg = RiskConfig(max_weight=0.3, max_gross=1.0, long_only=bool(trial % 2))
        rm = RiskManager(cfg, veto=lambda d: adversarial)
        limited = RiskManager(cfg).limits(target)
        final = rm.apply(target, None, 0)
        assert (final.abs() <= limited.abs() + 1e-12).all()
        assert final.abs().max() <= 0.3 + 1e-12 and final.abs().sum() <= 1.0 + 1e-12
        if cfg.long_only:
            assert (final >= 0).all()


def test_vol_target_scales_down_never_up():
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(0, 0.03, (100, 2)), columns=["A", "B"])  # ~48% vol each
    rm = RiskManager(RiskConfig(vol_target=0.10, long_only=False))
    w = rm.limits(pd.Series([0.5, 0.5], index=["A", "B"]), rets)
    assert w.abs().sum() < 0.5
    calm = pd.DataFrame(rng.normal(0, 0.001, (100, 2)), columns=["A", "B"])
    w2 = rm.limits(pd.Series([0.5, 0.5], index=["A", "B"]), calm)
    assert np.allclose(w2.values, [0.5, 0.5])


def test_kill_switch_flattens_then_resets():
    rm = RiskManager(RiskConfig(kill_drawdown=0.2, cooldown_days=3))
    assert not rm.on_close(1.0, 0)
    assert rm.on_close(0.75, 1)  # 25% drawdown
    assert (rm.apply(pd.Series([1.0], index=["A"]), None, 2) == 0).all()
    rm.on_close(0.75, 5)  # cooldown over
    assert (rm.apply(pd.Series([1.0], index=["A"]), None, 6) == 1).all()
