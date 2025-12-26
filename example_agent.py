"""
example_agent.py

Minimal smoke run to verify the package imports and the step loop executes.
This is not an evaluation harness; it just exercises the interfaces.
"""

from nupca3.agent import NUPCA3Agent
from nupca3.config import AgentConfig
from nupca3.types import EnvObs


def main() -> None:
    cfg = AgentConfig(D=16, B=4, fovea_blocks_per_step=2)
    agent = NUPCA3Agent(cfg)
    agent.reset(seed=0)

    # Partial observation: only dims 0 and 1 are observed.
    obs = EnvObs(x_partial={0: 0.1, 1: 0.2}, opp=0.0, danger=0.0)
    for _ in range(5):
        action, trace = agent.step(obs)
        print("action=", action, "t=", trace["t"], "rest=", trace["rest"], "h=", trace["h"])


if __name__ == "__main__":
    main()
