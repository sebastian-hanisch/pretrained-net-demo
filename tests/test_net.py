"""Das Netz: Vorwärtsrechnung von Hand, Ableitungen gegen Differenzenquotienten, Adam-Schritt von Hand, Lernen einer einfachen Abbildung."""

import numpy as np
import pytest

import ptn_net as N


def _shape():
    return N.Shape(window=6, covariates=3, horizon=4, blocks=2, width=5)


def _data(seed=0, n=7, sh=None):
    sh = sh or _shape()
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, sh.window)), rng.normal(size=(n, sh.covariates)), rng.normal(size=(n, sh.horizon))


def test_forward_by_hand_two_blocks():
    sh = N.Shape(window=2, covariates=1, horizon=2, blocks=2, width=2)
    p = {}
    for k in range(2):
        p[f"{k}.W1"] = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])
        p[f"{k}.b1"] = np.array([0.1, -0.1])
        p[f"{k}.W2"] = np.eye(2)
        p[f"{k}.b2"] = np.zeros(2)
        p[f"{k}.Wb"] = np.array([[1.0, 0.0], [0.0, 2.0]])
        p[f"{k}.bb"] = np.array([0.0, 0.5])
        p[f"{k}.Wf"] = np.array([[1.0, 1.0], [0.0, -1.0]])
        p[f"{k}.bf"] = np.array([0.0, 0.25])
    assert N.shape_of(p) == sh
    w, c = np.array([[1.0, 2.0]]), np.array([[2.0]])
    # Block 1: u = [1, 2, 2]; h1 = relu([1 + 1 + 0.1, 2 - 1 - 0.1]) = [2.1, 0.9]; h2 = h1; back = [2.1, 1.8 + 0.5] = [2.1, 2.3]; fc = [2.1, 2.1 - 0.9 + 0.25] = [2.1, 1.45]; r2 = [-1.1, -0.3]
    # Block 2: u = [-1.1, -0.3, 2]; h1 = relu([-1.1 + 1 + 0.1, -0.3 - 1 - 0.1]) = [0, 0]; h2 = 0; fc = bf = [0, 0.25]; Summe [2.1, 1.70]
    assert np.allclose(N.forward(p, w, c), [[2.1, 1.70]])


def test_residual_structure_matches_the_definition():
    sh = _shape()
    p = N.init(sh, seed=1)
    w, c, _ = _data(2)
    yhat, store = N.forward(p, w, c, cache=True)
    r, total = w, np.zeros((len(w), sh.horizon))
    for k in range(sh.blocks):
        u = np.concatenate([r, c], axis=1)
        h1 = np.maximum(u @ p[f"{k}.W1"] + p[f"{k}.b1"], 0)
        h2 = np.maximum(h1 @ p[f"{k}.W2"] + p[f"{k}.b2"], 0)
        assert np.allclose(store[k][0], u) and np.allclose(store[k][2], h2)
        total += h2 @ p[f"{k}.Wf"] + p[f"{k}.bf"]
        r = r - (h2 @ p[f"{k}.Wb"] + p[f"{k}.bb"])
    assert np.allclose(yhat, total)


@pytest.mark.parametrize("loss", ["l2", "l1"])
@pytest.mark.parametrize("blocks", [1, 3])
def test_gradients_match_finite_differences(loss, blocks):
    sh = N.Shape(6, 3, 4, blocks, 5)
    p = N.init(sh, seed=3)
    for k in p:                                           # von null verschiedene Biases, damit keine Relu-Knicke auf einer Stelle liegen
        p[k] = p[k] + 0.05 * np.random.default_rng(len(k)).normal(size=p[k].shape)
    w, c, y = _data(4, sh=sh)
    _, g = N.loss_and_grads(p, w, c, y, loss)
    worst = 0.0
    eps = 1e-6
    for k, arr in p.items():
        for idx in np.ndindex(*arr.shape):
            old = arr[idx]
            arr[idx] = old + eps
            up = N.loss_and_grads(p, w, c, y, loss)[0]
            arr[idx] = old - eps
            dn = N.loss_and_grads(p, w, c, y, loss)[0]
            arr[idx] = old
            worst = max(worst, abs((up - dn) / (2 * eps) - g[k][idx]))
    assert worst < 1e-6


def test_l1_loss_value_and_output_gradient_by_hand():
    sh = N.Shape(2, 0, 2, 1, 2)
    p = {"0.W1": np.eye(2), "0.b1": np.zeros(2), "0.W2": np.eye(2), "0.b2": np.zeros(2), "0.Wb": np.zeros((2, 2)), "0.bb": np.zeros(2), "0.Wf": np.eye(2), "0.bf": np.array([1.0, -1.0])}
    w, c, y = np.array([[1.0, 2.0]]), np.zeros((1, 0)), np.array([[0.0, 3.0]])
    val, g = N.loss_and_grads(p, w, c, y)
    # yhat = [1 + 1, 2 - 1] = [2, 1]; Fehler [2, -2]; MAE 2; d/d bf = sign / 2 = [0.5, -0.5]
    assert val == pytest.approx(2.0) and np.allclose(g["0.bf"], [0.5, -0.5]) and np.allclose(g["0.Wf"], [[0.5, -0.5], [1.0, -1.0]])


def test_first_adam_step_moves_every_parameter_by_lr_in_the_direction_of_the_gradient():
    sh = _shape()
    p = N.init(sh, seed=5)
    w, c, y = _data(6)
    _, g = N.loss_and_grads(p, w, c, y)
    lr = 1e-3
    q, _, _ = N.train(p, w, c, y, steps=1, batch=len(w), lr=lr, seed=0)
    for k in p:
        big = np.abs(g[k]) > 1e-6
        assert np.allclose((p[k] - q[k])[big], lr * np.sign(g[k][big]), atol=1e-6)


def test_training_leaves_the_input_untouched_and_is_deterministic():
    sh = _shape()
    p = N.init(sh, seed=7)
    before = N.copy(p)
    w, c, y = _data(8, n=40)
    a, ha, _ = N.train(p, w, c, y, steps=30, batch=8, lr=1e-2, seed=1)
    b, hb, _ = N.train(p, w, c, y, steps=30, batch=8, lr=1e-2, seed=1)
    assert all(np.array_equal(p[k], before[k]) for k in p) and all(np.array_equal(a[k], b[k]) for k in a) and ha == hb
    assert N.n_params(p) == sum(v.size for v in p.values())


def test_the_net_learns_a_linear_map_of_the_window():
    sh = N.Shape(8, 2, 3, 2, 16)
    rng = np.random.default_rng(9)
    w = rng.normal(size=(600, 8))
    c = rng.normal(size=(600, 2))
    A = rng.normal(size=(8, 3)) * 0.4
    y = w @ A + 0.3 * c[:, :1]
    p = N.init(sh, seed=2)
    start = N.mae(p, w, c, y)
    q, hist, _ = N.train(p, w, c, y, steps=1500, batch=64, lr=5e-3, seed=0)
    assert start > 0.5 and N.mae(q, w, c, y) < 0.15 * start and hist["train"][-1] < hist["train"][0]


def test_snapshots_and_history_lengths():
    sh = _shape()
    p = N.init(sh, seed=1)
    w, c, y = _data(3, n=32)
    q, hist, snaps = N.train(p, w, c, y, steps=20, batch=8, lr=1e-2, seed=0, steps_per_epoch=4, val=(w, c, y), snapshots=(0, 2, 5))
    assert len(hist["train"]) == 5 and len(hist["val"]) == 5 and set(snaps) == {0, 2, 5}
    assert all(np.array_equal(snaps[0][k], p[k]) for k in p) and all(np.array_equal(snaps[5][k], q[k]) for k in q) and not np.array_equal(snaps[2]["0.W1"], q["0.W1"])
