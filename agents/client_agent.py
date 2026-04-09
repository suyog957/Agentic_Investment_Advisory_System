# Simulated investor persona. Holds conversation history with the Advisor
# and responds in character based on the ClientProfile it was given.

from groq import Groq
from config import GROQ_API_KEY, MODEL_CHAT as MODEL
from schemas import ClientProfile


_SYSTEM_PROMPT_TEMPLATE = """
You are {name}, a {age}-year-old {occupation} seeking investment advice.

Your financial profile:
- Total investable assets: ${assets:,.0f}
- Annual income: ${income:,.0f}
- Risk tolerance: {risk}
- Investment horizon: {horizon} years
- Goals: {goals}
- Constraints: {constraints}
- Liquidity need: {liquidity}
- Existing investments: {existing}
- Additional context: {context}

Behavioral guidelines:
1. Respond as a real, slightly non-expert client would — you understand basic concepts but need guidance.
2. Be consistent with your profile in every message.
3. Ask natural follow-up questions when something is unclear.
4. Express your goals and concerns naturally (not in bullet-point format).
5. When you are genuinely satisfied with a recommendation and it addresses your goals,
   say something like "That sounds great, I'm happy to proceed with this plan" or similar.
6. Keep responses concise (2–4 sentences typically).
7. You are talking ONLY to your financial advisor — stay in character.
""".strip()


class ClientAgent:
    """Simulated client persona that responds to Advisor messages."""

    def __init__(self, profile: ClientProfile | None = None):
        self.profile = profile or ClientProfile()
        self._client = Groq(api_key=GROQ_API_KEY)
        self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            name=self.profile.name,
            age=self.profile.age,
            occupation=self.profile.occupation,
            assets=self.profile.assets,
            income=self.profile.annual_income,
            risk=self.profile.risk_tolerance,
            horizon=self.profile.investment_horizon_years,
            goals=", ".join(self.profile.goals),
            constraints=", ".join(self.profile.constraints),
            liquidity=self.profile.liquidity_need,
            existing=self.profile.existing_investments,
            context=self.profile.additional_context,
        )
        # History excludes the system message (injected at every call)
        self._history: list[dict] = []

    def respond(self, advisor_message: str) -> str:
        """Generate client response to an Advisor message."""
        self._history.append({"role": "user", "content": advisor_message})

        response = self._client.chat.completions.create(
            model=MODEL,
            max_tokens=512,
            messages=[
                {"role": "system", "content": self._system_prompt},
                *self._history,
            ],
        )
        client_reply = response.choices[0].message.content
        self._history.append({"role": "assistant", "content": client_reply})
        return client_reply

