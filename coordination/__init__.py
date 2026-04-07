"""
coordination/ — Global Coordination Layer for V18.

Provides:
  - Publish/Subscribe event bus (message_bus)
  - Multi-round consensus protocol (consensus_protocol)
  - Agent health supervisor (agent_supervisor)
  - Priority event scheduler (priority_scheduler)
  - Global finite state machine (state_machine)

This package acts as the nervous system that connects all V18
modules, ensuring thread-safe communication, fault tolerance,
and deterministic state transitions.
"""
