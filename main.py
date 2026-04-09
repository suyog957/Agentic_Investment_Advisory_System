"""
Entry point for the Agentic Investment Advisory System.

Usage:
    python main.py

Environment:
    Set ANTHROPIC_API_KEY in a .env file (copy from .env.example).
"""

from orchestrator import Orchestrator
from schemas import ClientProfile


def main():
    # Default client profile — modify or load from config to change the persona
    profile = ClientProfile(
        name="Sarah Chen",
        age=42,
        occupation="Senior Software Engineer",
        assets=350_000.0,
        annual_income=145_000.0,
        risk_tolerance="moderate",
        investment_horizon_years=20,
        goals=["retirement at age 65", "children's college education in 7-10 years"],
        constraints=["no cryptocurrency", "prefer ETFs over individual stocks", "no leveraged funds"],
        liquidity_need="low",
        existing_investments="401(k) with $120,000 balance, checking/savings $30,000 emergency fund",
        additional_context=(
            "Has two children aged 8 and 11. Currently maxing out 401(k) contributions. "
            "Has $350,000 in additional taxable brokerage cash to invest. "
            "Wants to retire at 65 (23 years away). "
            "Needs ~$80,000/year for college per child starting in 7 and 10 years."
        ),
    )

    session = Orchestrator(profile=profile)
    session.run()


if __name__ == "__main__":
    main()
