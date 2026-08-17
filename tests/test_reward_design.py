import math

from reward_design import compute_llm_reward, normalized_energy


def components(ssr=1.0, violation=0.0, power=0.0, energy=0.5):
    return {
        'positive_ssr': ssr, 'secrecy_violation': violation,
        'power_violation': power, 'energy_raw': energy,
        'energy_norm': energy,
    }


def test_energy_normalization_is_bounded():
    assert normalized_energy(-100, 0, 1) == 0.0
    assert normalized_energy(100, 0, 1) == 1.0


def test_balanced_secrecy_improves_reward():
    assert compute_llm_reward('llm_balanced', components(ssr=2))[1] > compute_llm_reward('llm_balanced', components(ssr=1))[1]


def test_balanced_energy_reduces_reward():
    assert compute_llm_reward('llm_balanced', components(energy=0))[1] > compute_llm_reward('llm_balanced', components(energy=1))[1]


def test_gated_energy_penalty_grows_with_secrecy():
    low = compute_llm_reward('llm_gated', components(ssr=0, energy=1))[0]
    high = compute_llm_reward('llm_gated', components(ssr=10, energy=1))[0]
    low_no_energy = compute_llm_reward('llm_gated', components(ssr=0, energy=0))[0]
    high_no_energy = compute_llm_reward('llm_gated', components(ssr=10, energy=0))[0]
    assert abs(low - low_no_energy) < abs(high - high_no_energy)


def test_ratio_energy_reduces_reward():
    assert compute_llm_reward('llm_ratio', components(energy=0))[1] > compute_llm_reward('llm_ratio', components(energy=1))[1]


def test_power_violation_reduces_all_rewards():
    for name in ('llm_balanced', 'llm_gated', 'llm_ratio'):
        assert compute_llm_reward(name, components(power=0))[1] > compute_llm_reward(name, components(power=1))[1]


def test_rewards_are_finite_and_bounded():
    for name in ('llm_balanced', 'llm_gated', 'llm_ratio'):
        raw, reward = compute_llm_reward(name, components(ssr=100, violation=10, power=10, energy=1))
        assert math.isfinite(raw) and math.isfinite(reward)
        assert -1.0 <= reward <= 1.0

