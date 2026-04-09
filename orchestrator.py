"""
Orchestrator — manages the Client ↔ Advisor conversation loop.

Flow:
  1. Advisor opens with a greeting.
  2. Client responds.
  3. Advisor processes (may internally delegate to Analyst).
  4. Repeat until termination condition is met.

Termination conditions:
  - Advisor calls terminate_conversation tool (client accepted / all resolved).
  - Max turn limit reached (safety cap).
"""

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from schemas import ClientProfile
from agents import ClientAgent, AnalystAgent, AdvisorAgent
from config import MAX_TURNS
from events import Event

console = Console()


def _print_turn(speaker: str, message: str, style: str = "white") -> None:
    """Pretty-print a conversation turn."""
    panel = Panel(
        message,
        title=f"[bold {style}]{speaker}[/bold {style}]",
        border_style=style,
        padding=(0, 1),
    )
    console.print(panel)
    console.print()


def _print_internal(label: str, detail: str) -> None:
    """Print an internal system event (not visible to either agent)."""
    console.print(f"[dim]  ⟳ {label}:[/dim] [italic dim]{detail}[/italic dim]")
    console.print()


class Orchestrator:
    """
    Top-level controller that runs the multi-agent investment advisory session.

    Architecture enforced here:
      Client → [only] → Advisor → [only] → Analyst (via delegate_to_analyst tool)
    """

    def __init__(self, profile: ClientProfile | None = None):
        self.profile = profile or ClientProfile()
        self.client_agent = ClientAgent(profile=self.profile)
        self.analyst_agent = AnalystAgent()
        self.advisor_agent = AdvisorAgent(
            profile=self.profile, analyst=self.analyst_agent
        )

    def run(self) -> None:
        """Execute the full advisory session."""
        console.print(Rule("[bold blue]Investment Advisory Session[/bold blue]"))
        console.print(
            f"[dim]Client profile: {self.profile.name}, age {self.profile.age}, "
            f"${self.profile.assets:,.0f} investable assets, "
            f"{self.profile.risk_tolerance} risk tolerance[/dim]"
        )
        console.print()

        # Step 1: Advisor opens the session
        _print_internal("Session", "Advisor generating opening greeting")
        advisor_message = self.advisor_agent.generate_opening()
        _print_turn("Advisor", advisor_message, style="green")

        turn = 0
        while turn < MAX_TURNS:
            turn += 1

            # Step 2: Client responds to Advisor
            _print_internal(
                f"Turn {turn}/{MAX_TURNS}",
                "Client formulating response",
            )
            client_message = self.client_agent.respond(advisor_message)
            _print_turn(self.profile.name + " (Client)", client_message, style="cyan")

            # Step 3: Advisor processes client message (may delegate to Analyst internally)
            _print_internal(
                f"Turn {turn}/{MAX_TURNS}",
                "Advisor processing (may consult Analyst internally)",
            )
            advisor_message = self.advisor_agent.process_client_message(client_message)
            _print_turn("Advisor", advisor_message, style="green")

            # Step 5: Check termination
            if self.advisor_agent.is_complete:
                console.print(
                    Rule(
                        f"[bold green]Session Complete[/bold green] — "
                        f"{self.advisor_agent.termination_reason}"
                    )
                )
                break
        else:
            console.print(
                Rule("[yellow]Session ended — maximum turn limit reached[/yellow]")
            )

        console.print("\n[dim]Disclaimer: This is an educational simulation and not real financial advice.[/dim]")

    def collect_events(self) -> list[Event]:
        """
        Run the full session and return every event as a list.
        Used by the Streamlit UI — no console output, pure data.
        """
        events: list[Event] = []

        def record(e: Event) -> None:
            events.append(e)

        # Wire advisor's event callback (it threads it through to the analyst)
        self.advisor_agent.on_event = record

        advisor_message = self.advisor_agent.generate_opening()
        events.append(Event("advisor", {"content": advisor_message}))

        turn = 0
        while turn < MAX_TURNS:
            turn += 1

            client_message = self.client_agent.respond(advisor_message)
            events.append(Event("client", {"content": client_message, "name": self.profile.name}))

            advisor_message = self.advisor_agent.process_client_message(client_message)
            events.append(Event("advisor", {"content": advisor_message}))

            if self.advisor_agent.is_complete:
                events.append(Event("session_complete", {
                    "reason": self.advisor_agent.termination_reason,
                    "turns": turn,
                }))
                break
        else:
            events.append(Event("session_complete", {
                "reason": "Maximum turn limit reached.",
                "turns": turn,
            }))

        return events
