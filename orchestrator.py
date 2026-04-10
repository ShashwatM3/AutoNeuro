"""
Main loop / orchestrator for the multi-agent AutoResearch-style ML pipeline.

Per README: manages git state, delegates runs to wrapper.sh, reads state.json and
results.tsv, polls HUMAN_INSTRUCTION.txt, and implements LOOP FOREVER with agent
escalation. This module is still a stub; run training via ./wrapper.sh.
"""


def main() -> None:
    print("orchestrator stub — invoke ./wrapper.sh for a training run")


if __name__ == "__main__":
    main()
