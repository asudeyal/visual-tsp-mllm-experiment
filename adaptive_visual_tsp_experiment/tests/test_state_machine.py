from src.controller.state_machine import EscapeStateMachine


def test_first_stagnation_hybrid_second_restart():
    machine = EscapeStateMachine()
    assert machine.action_for_stagnation() == "hybrid"
    assert machine.action_for_stagnation() == "restart"
    machine.mark_restart()
    assert machine.action_for_stagnation() == "hybrid"
