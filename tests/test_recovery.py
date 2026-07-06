from self_healing_vio.health import VioHealthEstimate
from self_healing_vio.recovery import RecoveryPolicy


def test_recovery_policy_continues_when_healthy():
    decision = RecoveryPolicy().decide(VioHealthEstimate(0.9, 'healthy', 0.1))
    assert decision.action == 'continue_nominal_tracking'
    assert decision.priority == 1


def test_recovery_policy_adapts_when_warning():
    decision = RecoveryPolicy().decide(VioHealthEstimate(0.4, 'warning', 0.6))
    assert decision.action == 'increase_inertial_weight'
    assert decision.priority == 2


def test_recovery_policy_recovers_when_critical():
    decision = RecoveryPolicy().decide(VioHealthEstimate(0.1, 'critical', 0.9))
    assert decision.action == 'trigger_relocalization'
    assert decision.priority == 3
