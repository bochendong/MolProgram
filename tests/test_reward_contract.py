from molprogram.rewards import INELIGIBLE_FLOOR, hard_boundary_reward


def test_ineligible_edit_stays_at_common_floor():
    details = {
        "valid": True,
        "copy": False,
        "source_similarity": 0.64,
        "canonical": True,
        "mean_satisfaction": 1.0,
        "bottleneck": 1.0,
        "property_fraction": 1.0,
        "property_strict": True,
        "strict": False,
    }
    assert hard_boundary_reward({}, details, "edit") == INELIGIBLE_FLOOR


def test_eligible_strict_edit_clears_floor():
    details = {
        "valid": True,
        "copy": False,
        "source_similarity": 0.80,
        "canonical": True,
        "mean_satisfaction": 1.0,
        "bottleneck": 1.0,
        "property_fraction": 1.0,
        "property_strict": True,
        "strict": True,
    }
    assert hard_boundary_reward({}, details, "edit") > 0.0
