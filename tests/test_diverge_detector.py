import random

import pytest

from levanter.trainer import DivergenceDetector


@pytest.fixture
def det() -> DivergenceDetector:
    return DivergenceDetector(threshold=0.2, patience=3)


# Test setup and teardown
def test_setup_teardown(det: DivergenceDetector):
    assert det is not None
    assert det.previous_losses == []
    assert det.threshold == 0.2
    assert det.patience == 3
    assert det.diverged is False


def normal_random_walk(start, delta_mu, std, seed=42, n=50):
    r = random.Random(seed)
    res = [float(start)]
    for i in range(n - 1):
        res.append(res[-1] + r.normalvariate(delta_mu, std))
    return res


# Test initialization with invalid parameters
def test_init_invalid_parameters():
    with pytest.raises(AssertionError):
        DivergenceDetector(threshold=-0.1, patience=3)
    with pytest.raises(AssertionError):
        DivergenceDetector(threshold=0.2, patience=0)


# Test update method
def test_update_method(det: DivergenceDetector):
    for i in range(1, 5):  # Adding losses from 1 to 4
        det.update(i)
        assert det.previous_losses == list(range(1, i + 1))[-det.patience :]


def test_check_divergence_no_data(det: DivergenceDetector):
    # No data available for comparison, should not diverge
    new_loss: float = 5.0
    assert det.check_divergence(new_loss) is False


@pytest.mark.parametrize("new_loss", [1.0, 2.0, 3.0])
def test_check_divergence_with_divergent_data(det: DivergenceDetector, new_loss: float):
    # Adding losses to fill patience window and check for divergence
    for _ in range(det.patience - 2):  # Add three losses less than threshold difference
        det.update(_)

    assert det.check_divergence(new_loss) is False  # Not enough data, no divergence expected


@pytest.mark.parametrize("delta_mu,diverges", [(1.0, True), (0.1, True), (-0.1, False), (-1.0, False)])
@pytest.mark.parametrize("seed", list(range(50)))
def test_divergence_against_random_walks(det: DivergenceDetector, delta_mu, diverges, seed):
    # Adding losses to fill patience window and check for divergence
    det.threshold = 0.2
    det.patience = 20
    walk = normal_random_walk(start=0, delta_mu=delta_mu, std=0.1, seed=seed, n=100)
    res = []
    for x in walk:
        res.append(det.check_divergence(x))
    print(list(zip(walk, res)))
    assert any(res) is diverges, f"{diverges}, {res}"


@pytest.mark.parametrize("new_loss", [2.0, 3.0])
def test_check_divergence_with_data_and_divergence(det: DivergenceDetector, new_loss: float):
    # Adding losses to fill patience window and check for divergence
    for _ in range(det.patience - 1):  # Add two losses less than threshold difference
        det.update(_)

    assert det.check_divergence(new_loss) is True  # Divergence detected


@pytest.mark.parametrize("new_loss", [1.0, 1.1])
def test_no_divergence(det: DivergenceDetector, new_loss: float):
    # Adding losses to fill patience window and check for divergence
    for _ in [3, 2, 1, 1]:  # Add two losses less than threshold difference
        det.update(_)

    assert det.check_divergence(new_loss) is False  # Divergence not yet detected
