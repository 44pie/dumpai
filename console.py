"""
DumpAI Console - Rich Terminal Output

Provides structured, beautiful console output inspired by hackingBuddyGPT.
Shows AI reasoning, round-based progress, and color-coded phases.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

try:
    from rich.console import Console as RichConsole
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.box import ROUNDED, SIMPLE
    from rich.style import Style
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class LogLevel(Enum):
    INFO = "info"
    AI = "ai"
    SUCCESS = "success"
    ERROR = "error"
    WARN = "warn"
    DEBUG = "debug"
    PHASE = "phase"
    ROUND = "round"


class DumpAIConsole:
    """
    Rich console for DumpAI with structured AI reasoning output.
    
    Verbosity levels:
    - 0: Minimal (only phases and results)
    - 1 (-v): Show AI reasoning summaries
    - 2 (-vv): Show full AI reasoning and all decisions
    """
    
    COLORS = {
        LogLevel.INFO: "white",
        LogLevel.AI: "cyan",
        LogLevel.SUCCESS: "green",
        LogLevel.ERROR: "red",
        LogLevel.WARN: "yellow",
        LogLevel.DEBUG: "dim white",
        LogLevel.PHASE: "bold magenta",
        LogLevel.ROUND: "bold blue",
    }
    
    SYMBOLS = {
        LogLevel.INFO: "+",
        LogLevel.AI: "*",
        LogLevel.SUCCESS: "✓",
        LogLevel.ERROR: "!",
        LogLevel.WARN: "⚠",
        LogLevel.DEBUG: ".",
        LogLevel.PHASE: "═",
        LogLevel.ROUND: "►",
    }
    
    def __init__(self, verbosity: int = 0, debug_file: str = None):
        self.verbosity = verbosity
        self.debug_file = None
        self.round_num = 0
        
        if debug_file:
            self.debug_file = open(debug_file, "w", buffering=1)
            self._debug(f"=== DumpAI Debug Log Started: {datetime.now().isoformat()} ===")
        
        if RICH_AVAILABLE:
            self.console = RichConsole()
            self.use_rich = True
        else:
            self.console = None
            self.use_rich = False
    
    def _debug(self, msg: str):
        """Write to debug log file."""
        if self.debug_file:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.debug_file.write(f"[{timestamp}] {msg}\n")
    
    def _timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")
    
    def log(self, msg: str, level: LogLevel = LogLevel.INFO):
        """Log a message with appropriate styling."""
        self._debug(f"[{level.value}] {msg}")
        
        if level == LogLevel.DEBUG and self.verbosity < 2:
            return
        
        timestamp = self._timestamp()
        symbol = self.SYMBOLS.get(level, "+")
        
        if self.use_rich:
            color = self.COLORS.get(level, "white")
            self.console.print(f"[dim]{timestamp}[/dim] [{color}][{symbol}][/{color}] {msg}")
        else:
            print(f"[{timestamp}] [{symbol}] {msg}")
    
    def phase(self, title: str):
        """Display a phase header."""
        self._debug(f"\n{'='*60}\n  PHASE: {title}\n{'='*60}")
        
        if self.use_rich:
            self.console.print()
            self.console.print(Panel(
                Text(title, style="bold white"),
                border_style="magenta",
                padding=(0, 2),
            ))
        else:
            print()
            print("=" * 60)
            print(f"  {title}")
            print("=" * 60)
    
    def round_start(self, round_num: int, action: str, params: Dict = None):
        """Display round start with action."""
        self.round_num = round_num
        
        if self.verbosity < 1:
            return
        
        self._debug(f"\n--- ROUND {round_num}: {action} ---")
        
        if self.use_rich:
            self.console.print()
            header = f"ROUND {round_num}"
            self.console.print(f"[bold blue]═══ {header} {'═' * (50 - len(header))}[/bold blue]")
            self.console.print(f"[cyan]ACTION:[/cyan] {action}")
            if params and self.verbosity >= 2:
                for k, v in params.items():
                    self.console.print(f"  [dim]{k}:[/dim] {v}")
        else:
            print()
            print(f"═══ ROUND {round_num} {'═' * 45}")
            print(f"ACTION: {action}")
    
    def round_result(self, success: bool, summary: str, data: Any = None):
        """Display round result."""
        if self.verbosity < 1:
            return
        
        status = "SUCCESS" if success else "FAILED"
        self._debug(f"RESULT: {status} - {summary}")
        
        if self.use_rich:
            color = "green" if success else "red"
            symbol = "✓" if success else "✗"
            self.console.print(f"[{color}]RESULT: {symbol} {summary}[/{color}]")
            
            if data and self.verbosity >= 2:
                if isinstance(data, list):
                    self.console.print(f"  [dim]Data: {len(data)} items[/dim]")
                elif isinstance(data, dict):
                    self.console.print(f"  [dim]Data: {list(data.keys())[:5]}...[/dim]")
        else:
            symbol = "+" if success else "!"
            print(f"RESULT: [{symbol}] {summary}")
    
    def ai_thinking(self, title: str = "AI REASONING"):
        """Start an AI thinking block - returns context for 'with' statement."""
        if self.verbosity < 1:
            return _NoOpContext()
        
        return _AIThinkingContext(self, title)
    
    def ai_reason(self, reasoning: str, decision: str = None, 
                  confidence: float = None, alternatives: List[str] = None):
        """Display AI reasoning in a structured panel."""
        if self.verbosity < 1:
            return
        
        self._debug(f"AI_REASONING: {reasoning}")
        if decision:
            self._debug(f"AI_DECISION: {decision}")
        
        if self.use_rich:
            content = Text()
            
            if reasoning:
                content.append(reasoning + "\n", style="white")
            
            if decision:
                content.append(f"\n→ Decision: ", style="bold cyan")
                content.append(decision, style="white")
            
            if confidence is not None and self.verbosity >= 2:
                content.append(f"\n  Confidence: ", style="dim")
                conf_color = "green" if confidence > 0.7 else "yellow" if confidence > 0.4 else "red"
                content.append(f"{confidence:.0%}", style=conf_color)
            
            if alternatives and self.verbosity >= 2:
                content.append(f"\n  Alternatives: ", style="dim")
                content.append(", ".join(alternatives), style="dim")
            
            self.console.print(Panel(
                content,
                title="[bold cyan]AI THINKING[/bold cyan]",
                border_style="cyan",
                padding=(0, 1),
            ))
        else:
            print("┌─ AI THINKING ─────────────────────┐")
            for line in reasoning.split('\n'):
                print(f"│ {line:<35} │")
            if decision:
                print(f"│ → {decision:<33} │")
            print("└───────────────────────────────────┘")
    
    def table_priority(self, tables: List[Dict]):
        """Display prioritized tables in a nice table."""
        if self.verbosity < 1 or not tables:
            return
        
        if self.use_rich:
            table = Table(title="Prioritized Tables", box=ROUNDED)
            table.add_column("Score", style="cyan", width=6)
            table.add_column("Table", style="white")
            table.add_column("Category", style="yellow")
            table.add_column("Reason", style="dim")
            
            for t in tables[:10]:
                score = t.get('score', 0)
                score_style = "green" if score > 0.8 else "yellow" if score > 0.5 else "red"
                table.add_row(
                    f"[{score_style}]{score:.2f}[/{score_style}]",
                    t.get('table', '?'),
                    t.get('category', '?'),
                    t.get('reason', '')[:30]
                )
            
            self.console.print(table)
        else:
            print("\nPrioritized Tables:")
            for t in tables[:10]:
                print(f"  {t.get('score', 0):.2f} {t.get('table', '?')}: {t.get('reason', '')}")
    
    def extraction_progress(self, current: int, total: int, table: str):
        """Show extraction progress."""
        if self.use_rich and self.verbosity >= 1:
            pct = (current / total * 100) if total > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            self.console.print(f"  [{bar}] {current}/{total} - {table}", end="\r")
    
    def summary_table(self, stats: Dict):
        """Display final summary as a table."""
        if self.use_rich:
            table = Table(title="Extraction Summary", box=SIMPLE)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in stats.items():
                if isinstance(value, float):
                    value = f"{value:.1f}"
                table.add_row(key.replace('_', ' ').title(), str(value))
            
            self.console.print()
            self.console.print(table)
        else:
            print("\n=== Summary ===")
            for key, value in stats.items():
                print(f"  {key}: {value}")
    
    def error_panel(self, error: str, suggestion: str = None):
        """Display error with optional suggestion."""
        self._debug(f"ERROR: {error}")
        
        if self.use_rich:
            content = Text(error, style="red")
            if suggestion:
                content.append(f"\n\n→ Suggestion: ", style="yellow")
                content.append(suggestion, style="white")
            
            self.console.print(Panel(
                content,
                title="[bold red]ERROR[/bold red]",
                border_style="red",
                padding=(0, 1),
            ))
        else:
            print(f"[!] ERROR: {error}")
            if suggestion:
                print(f"    → {suggestion}")
    
    def injection_analysis(self, analysis: Dict):
        """Display injection analysis results."""
        if self.verbosity < 1:
            self.log(f"Injection: {analysis.get('injection_type', 'unknown')}", LogLevel.AI)
            return
        
        self._debug(f"INJECTION_ANALYSIS: {analysis}")
        
        if self.use_rich:
            content = Text()
            
            inj_type = analysis.get('injection_type', 'unknown')
            is_slow = analysis.get('is_slow', False)
            
            content.append("Type: ", style="dim")
            content.append(f"{inj_type}\n", style="bold white")
            
            content.append("Speed: ", style="dim")
            speed_style = "red" if is_slow else "green"
            content.append(f"{'SLOW' if is_slow else 'FAST'}\n", style=speed_style)
            
            content.append("DBMS: ", style="dim")
            content.append(f"{analysis.get('dbms', 'unknown')}\n", style="white")
            
            if analysis.get('waf_detected'):
                content.append("\n⚠ WAF DETECTED\n", style="bold yellow")
                tampers = analysis.get('tampers_suggested', [])
                if tampers:
                    content.append(f"  Suggested tampers: {', '.join(tampers)}", style="dim")
            
            strategy = analysis.get('recommended_strategy', 'unknown')
            content.append(f"\n\n→ Strategy: ", style="cyan")
            content.append(strategy, style="bold white")
            
            if analysis.get('reasoning'):
                content.append(f"\n  {analysis['reasoning']}", style="dim")
            
            self.console.print(Panel(
                content,
                title="[bold cyan]INJECTION ANALYSIS[/bold cyan]",
                border_style="cyan",
            ))
        else:
            print(f"Injection Type: {analysis.get('injection_type', 'unknown')}")
            print(f"Speed: {'SLOW' if analysis.get('is_slow') else 'FAST'}")
            print(f"Strategy: {analysis.get('recommended_strategy', 'unknown')}")
    
    def close(self):
        """Close debug file if open."""
        if self.debug_file:
            self._debug(f"=== DumpAI Debug Log Ended: {datetime.now().isoformat()} ===")
            self.debug_file.close()


class _NoOpContext:
    """No-op context manager for when verbosity is too low."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


class _AIThinkingContext:
    """Context manager for AI thinking block."""
    def __init__(self, console: DumpAIConsole, title: str):
        self.console = console
        self.title = title
    
    def __enter__(self):
        if self.console.use_rich:
            self.console.console.print(f"[dim cyan]┌─ {self.title} ─[/dim cyan]")
        return self
    
    def __exit__(self, *args):
        if self.console.use_rich:
            self.console.console.print(f"[dim cyan]└{'─' * (len(self.title) + 4)}[/dim cyan]")
