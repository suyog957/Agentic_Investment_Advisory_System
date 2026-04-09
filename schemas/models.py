from pydantic import BaseModel, Field


class ClientProfile(BaseModel):
    """Simulated client persona."""
    name: str = "Sarah Chen"
    age: int = 42
    occupation: str = "Senior Software Engineer"
    assets: float = 350_000.0
    annual_income: float = 145_000.0
    risk_tolerance: str = "moderate"   # conservative | moderate | aggressive
    investment_horizon_years: int = 20
    goals: list[str] = Field(default_factory=lambda: ["retirement", "children's education"])
    constraints: list[str] = Field(default_factory=lambda: ["no cryptocurrency", "prefer ETFs", "no individual stocks"])
    liquidity_need: str = "low"        # low | medium | high
    existing_investments: str = "401(k) with $120K, checking/savings $30K"
    additional_context: str = (
        "Has two children aged 8 and 11. Wants retirement at 65. "
        "Children's college funds needed in ~7-10 years. "
        "Currently maxing 401(k). Looking to invest remaining assets wisely."
    )


class AnalystTask(BaseModel):
    """Task sent from Advisor to Analyst."""
    objective: str
    research_questions: list[str]
    client_profile_summary: str | None = None


class AnalystReport(BaseModel):
    """Structured research output returned by Analyst."""
    market_summary: str
    recommendations: list[str]
    risks: list[str]
    etf_suggestions: list[dict] = Field(default_factory=list)
    allocation_breakdown: dict | None = None
    sources: list[str] = Field(default_factory=list)
