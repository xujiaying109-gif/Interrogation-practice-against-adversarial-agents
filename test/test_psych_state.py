import pytest
from psych_state import MentalStateMachine
from personality import CalmPersonality, NervousPersonality
from config import config

def test_mental_state_initialization():
    machine = MentalStateMachine(total_rounds=10, personality=CalmPersonality())
    assert machine.defense == 75.0
    assert machine.stress == 25.0

def test_mental_state_update_confession():
    machine = MentalStateMachine(total_rounds=10, personality=NervousPersonality())
    machine.defense = 6.0
    machine.stress = 96.0
    
    assert machine.defense < 10.0
