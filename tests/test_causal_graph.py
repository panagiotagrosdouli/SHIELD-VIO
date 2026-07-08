from self_healing_vio.diagnosis import FailureCausalGraph


def test_causal_graph_outputs_bounded_risks():
    graph = FailureCausalGraph()
    risks = graph.from_health_scores({"motion_blur": 0.2, "feature_collapse": 0.4})
    assert "localization_failure" in risks
    assert all(0.0 <= value <= 1.0 for value in risks.values())
