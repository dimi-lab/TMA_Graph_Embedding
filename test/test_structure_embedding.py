import numpy as np
import networkx as nx

from src.embeddings import structure_embedding


def test_fastrp_shape_with_weights():
    G = nx.path_graph(5)  # 5 nodes
    E = structure_embedding(G, method="fastrp", dim=8, q=3, weights=[1.0, 0.5, 0.25])
    assert isinstance(E, np.ndarray)
    assert E.shape == (5, 8)             # linear combo -> dim stays 8
    assert np.isfinite(E).all()
    # has variance
    assert E.std() > 0


def test_fastrp_shape_concat_when_weights_none():
    G = nx.path_graph(7)
    E = structure_embedding(G, method="fastrp", dim=8, q=2, weights=None)  # concatenate
    assert E.shape == (7, 16)            # 8 * q
    assert np.isfinite(E).all()


def test_fastrp_deterministic_same_inputs():
    G = nx.cycle_graph(10)
    E1 = structure_embedding(G, method="fastrp", dim=16, q=3, weights=[1.0, 0.5, 0.25])
    E2 = structure_embedding(G, method="fastrp", dim=16, q=3, weights=[1.0, 0.5, 0.25])
    assert np.allclose(E1, E2, atol=1e-8)


def test_fastrp_input_matrix_variants():
    G = nx.path_graph(8)
    E_adj = structure_embedding(G, method="fastrp", dim=8, q=3, weights=[1.0, 0.5, 0.25], input_matrix="adj")
    E_trans = structure_embedding(G, method="fastrp", dim=8, q=3, weights=[1.0, 0.5, 0.25], input_matrix="trans")
    assert E_adj.shape == E_trans.shape == (8, 8)
    # Usually different; allow rare numerical equality with a tolerant check
    assert not np.allclose(E_adj, E_trans, atol=1e-10)


def test_fastrp_raises_on_weights_length_mismatch():
    G = nx.path_graph(4)
    try:
        structure_embedding(G, method="fastrp", dim=8, q=3, weights=[1.0, 0.5])  # len!=q
        assert False, "Expected ValueError due to weights length mismatch"
    except ValueError:
        pass
