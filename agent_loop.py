"""
DumpAI Agent Loop - Autonomous Reason→Act→Observe Cycle

Inspired by hackingBuddyGPT's perform_round architecture.
Each round: AI reasons about state, decides action, executes, observes result.
"""
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from enum import Enum

try:
    from .planner import Planner, Decision, Observation, ActionType
    from .memory import Memory
except ImportError:
    from planner import Planner, Decision, Observation, ActionType
    from memory import Memory


class AgentPhase(Enum):
    """Agent execution phases."""
    INIT = "init"
    INJECTION_ANALYSIS = "injection_analysis"
    DATABASE_DISCOVERY = "database_discovery"
    TABLE_DISCOVERY = "table_discovery"
    TABLE_PRIORITIZATION = "table_prioritization"
    COLUMN_ANALYSIS = "column_analysis"
    EXTRACTION = "extraction"
    COMPLETION = "completion"
    ABORTED = "aborted"


@dataclass
class RoundResult:
    """Result of a single agent round."""
    round_num: int
    phase: AgentPhase
    decision: Decision
    observation: Observation
    phase_changed: bool = False
    should_continue: bool = True


class AgentLoop:
    """
    Autonomous agent loop following hackingBuddyGPT pattern.
    
    Architecture:
    1. UseCase (DumpAI) creates AgentLoop with tools and config
    2. AgentLoop runs perform_round() until complete or max_rounds
    3. Each round: Planner decides -> Tool executes -> Agent observes
    4. Memory tracks all state for AI context
    
    Key differences from old DumpAgent:
    - AI makes decisions at EVERY step (not just CMS detection)
    - Adapts to errors/timeouts in real-time
    - Dynamic table prioritization by AI
    - No hardcoded phase transitions
    """
    
    BANNER = """
░█▀▄░█░█░█▄█░█▀█░█▀█░▀█▀
░█░█░█░█░█░█░█▀▀░█▀█░░█░
░▀▀░░▀▀▀░▀░▀░▀░░░▀░▀░▀▀▀

  AI-Powered Autonomous Agent
  v3.0 - Full AI Loop
"""
    
    def __init__(self, 
                 tools: Dict[str, Callable],
                 config: Dict,
                 categories: List[str],
                 output_dir: str = "dumpai_out",
                 max_rounds: int = 100,
                 verbose: bool = False):
        """
        Initialize the agent loop.
        
        Args:
            tools: Dict of tool_name -> tool executor function
            config: SQLMap configuration (base_cmd, database, etc)
            categories: Data categories to extract
            output_dir: Where to save results
            max_rounds: Max rounds before forced stop
            verbose: Print detailed output
        """
        self.tools = tools
        self.config = config
        self.categories = categories
        self.output_dir = output_dir
        self.max_rounds = max_rounds
        self.verbose = verbose
        
        self.memory = Memory()
        self.memory.current_database = config.get("database", "")
        
        self.planner = Planner(verbosity=1 if verbose else 0)
        
        self.phase = AgentPhase.INIT
        self.round_num = 0
        self.start_time = None
        
        self.pending_tables: List[Dict] = []
        self.current_table: Optional[str] = None
        self.retry_count = 0
        self.max_retries = 3
    
    def _log(self, msg: str, level: str = "INFO"):
        """Log with timestamp and symbol."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        symbols = {
            "INFO": "+", "AI": "*", "ACTION": ">",
            "SUCCESS": "+", "ERROR": "!", "PHASE": "="
        }
        symbol = symbols.get(level, "+")
        print(f"[{timestamp}] [{symbol}] {msg}")
    
    def _phase_header(self, title: str):
        """Print phase header."""
        print()
        print("=" * 60)
        print(f"  {title}")
        print("=" * 60)
    
    def run(self) -> Dict:
        """
        Main entry point - runs the agent until completion.
        
        Returns:
            Summary dict with results
        """
        print(self.BANNER)
        self.start_time = time.time()
        
        self._phase_header("AUTONOMOUS AI AGENT STARTED")
        self._log(f"Categories: {', '.join(self.categories)}")
        self._log(f"Max rounds: {self.max_rounds}")
        self._log(f"Output: {self.output_dir}")
        
        while self.round_num < self.max_rounds:
            result = self.perform_round()
            
            if not result.should_continue:
                if result.phase == AgentPhase.COMPLETION:
                    self._log("Mission accomplished!", "SUCCESS")
                elif result.phase == AgentPhase.ABORTED:
                    self._log("Agent aborted due to critical failure", "ERROR")
                break
            
            self.round_num += 1
        
        if self.round_num >= self.max_rounds:
            self._log(f"Max rounds ({self.max_rounds}) reached", "ERROR")
        
        return self._finalize()
    
    def perform_round(self) -> RoundResult:
        """
        Execute a single agent round (like hackingBuddyGPT.perform_round).
        
        Pattern:
        1. REASON: Planner analyzes state and decides action
        2. ACT: Execute the chosen tool
        3. OBSERVE: Parse result and update memory
        4. ADAPT: Determine if strategy needs to change
        
        Returns:
            RoundResult with decision, observation, and continuation flag
        """
        self.round_num += 1
        
        if self.verbose:
            self._log(f"Round {self.round_num} | Phase: {self.phase.value}", "INFO")
        
        context = self.memory.get_context_for_ai()
        last_obs = self._get_last_observation()
        
        decision = self.planner.decide_next_action(
            memory_context=context,
            last_observation=last_obs,
            phase=self.phase.value
        )
        
        if self.verbose:
            self._log(f"AI Decision: {decision.action.value}", "AI")
            self._log(f"Reasoning: {decision.reasoning}", "AI")
        
        observation = self._execute_action(decision)
        
        self._update_memory(decision, observation)
        
        phase_changed, should_continue = self._evaluate_and_adapt(decision, observation)
        
        return RoundResult(
            round_num=self.round_num,
            phase=self.phase,
            decision=decision,
            observation=observation,
            phase_changed=phase_changed,
            should_continue=should_continue
        )
    
    def _get_last_observation(self) -> Optional[Observation]:
        """Get last observation from memory."""
        if self.memory.history:
            last = self.memory.history[-1]
            return Observation(
                action=last.tool,
                success=last.success,
                data=last.result,
                execution_time=last.execution_time
            )
        return None
    
    def _execute_action(self, decision: Decision) -> Observation:
        """Execute the decided action using appropriate tool."""
        action = decision.action
        params = decision.params
        
        start = time.time()
        
        if action == ActionType.COMPLETE:
            return Observation(
                action="complete",
                success=True,
                insights={"status": "extraction_complete"}
            )
        
        if action == ActionType.ABORT:
            return Observation(
                action="abort",
                success=False,
                error="Agent decided to abort"
            )
        
        tool_name = action.value
        tool = self.tools.get(tool_name)
        
        if not tool:
            return Observation(
                action=tool_name,
                success=False,
                error=f"Tool not found: {tool_name}",
                execution_time=time.time() - start
            )
        
        try:
            if self.verbose:
                self._log(f"Executing: {tool_name}", "ACTION")
            
            result = tool(**params)
            
            return Observation(
                action=tool_name,
                success=result.success if hasattr(result, 'success') else True,
                data=result.data if hasattr(result, 'data') else result,
                raw_output=result.raw_output if hasattr(result, 'raw_output') else "",
                error=result.error if hasattr(result, 'error') else "",
                execution_time=time.time() - start,
                insights=self._extract_insights(tool_name, result)
            )
            
        except Exception as e:
            return Observation(
                action=tool_name,
                success=False,
                error=str(e),
                execution_time=time.time() - start
            )
    
    def _extract_insights(self, tool_name: str, result: Any) -> Dict:
        """Extract actionable insights from tool result."""
        insights = {}
        
        if tool_name == "enumerate_dbs" and hasattr(result, 'data') and result.data:
            system_dbs = {'information_schema', 'mysql', 'performance_schema', 'sys'}
            user_dbs = [db for db in result.data if db.lower() not in system_dbs]
            insights["databases_found"] = len(result.data)
            insights["user_databases"] = user_dbs
        
        elif tool_name == "enumerate_tables" and hasattr(result, 'data') and result.data:
            insights["tables_found"] = len(result.data)
        
        elif tool_name == "search_tables" and hasattr(result, 'data') and result.data:
            insights["matched_tables"] = len(result.data)
        
        elif tool_name in ("dump_table", "dump_columns") and hasattr(result, 'data') and result.data:
            insights["rows_extracted"] = len(result.data)
        
        if hasattr(result, 'raw_output') and result.raw_output:
            raw = result.raw_output.lower()
            if "time-based blind" in raw:
                insights["injection_type"] = "time_based"
            elif "boolean-based blind" in raw:
                insights["injection_type"] = "boolean_blind"
            elif "error-based" in raw:
                insights["injection_type"] = "error_based"
            elif "union" in raw:
                insights["injection_type"] = "union"
            
            if any(x in raw for x in ["waf", "blocked", "forbidden"]):
                insights["waf_detected"] = True
        
        return insights
    
    def _update_memory(self, decision: Decision, observation: Observation):
        """Update agent memory with action result."""
        self.memory.add_action(
            tool=observation.action,
            params=decision.params,
            result=observation.data,
            success=observation.success,
            execution_time=observation.execution_time
        )
        
        if not observation.success and observation.error:
            self.memory.add_error(
                observation.action,
                observation.error,
                decision.params
            )
        
        if observation.success and observation.data:
            action = decision.action
            
            if action == ActionType.ENUMERATE_DBS:
                self.memory.databases = observation.data
                if observation.insights.get("user_databases"):
                    db = observation.insights["user_databases"][0]
                    self.memory.current_database = db
            
            elif action == ActionType.ENUMERATE_TABLES:
                self.memory.tables = observation.data
            
            elif action == ActionType.SEARCH_TABLES:
                if not self.memory.tables:
                    self.memory.tables = []
                for item in observation.data:
                    if '.' in item:
                        table = item.split('.')[-1]
                    else:
                        table = item
                    if table not in self.memory.tables:
                        self.memory.tables.append(table)
            
            elif action in (ActionType.DUMP_TABLE, ActionType.DUMP_COLUMNS):
                if isinstance(observation.data, list):
                    for cat in self.categories:
                        self.memory.add_extracted_data(
                            cat, observation.data,
                            source_table=self.current_table or ""
                        )
        
        if observation.insights.get("injection_type"):
            self.memory.database_type = observation.insights["injection_type"]
    
    def _evaluate_and_adapt(self, decision: Decision, 
                            observation: Observation) -> tuple:
        """
        Evaluate result and adapt strategy if needed.
        
        Returns:
            (phase_changed: bool, should_continue: bool)
        """
        if decision.action == ActionType.COMPLETE:
            self.phase = AgentPhase.COMPLETION
            return True, False
        
        if decision.action == ActionType.ABORT:
            self.phase = AgentPhase.ABORTED
            return True, False
        
        if not observation.success:
            self.retry_count += 1
            
            if self.retry_count >= self.max_retries:
                self._log(f"Max retries reached for {observation.action}", "ERROR")
                
                if self.current_table:
                    self._log(f"Skipping table: {self.current_table}", "INFO")
                    self.pending_tables = [
                        t for t in self.pending_tables 
                        if t.get("table") != self.current_table
                    ]
                    self.current_table = None
                    self.retry_count = 0
                    return False, True
                
                adaptation = self.planner.adapt_to_error(
                    error=observation.error,
                    context=self.memory.get_context_for_ai(),
                    failed_action=observation.action
                )
                
                if adaptation.get("should_retry"):
                    self._log(f"AI suggests retry: {adaptation.get('reasoning', '')}", "AI")
                    self.retry_count = 0
                    return False, True
                
                self._log("AI suggests abort", "AI")
                self.phase = AgentPhase.ABORTED
                return True, False
            
            return False, True
        
        self.retry_count = 0
        
        old_phase = self.phase
        self._update_phase(decision, observation)
        
        return self.phase != old_phase, True
    
    def _update_phase(self, decision: Decision, observation: Observation):
        """Update agent phase based on completed action."""
        if self.phase == AgentPhase.INIT:
            self.phase = AgentPhase.INJECTION_ANALYSIS
            self._phase_header("PHASE 1: INJECTION ANALYSIS")
        
        elif self.phase == AgentPhase.INJECTION_ANALYSIS:
            if observation.insights.get("injection_type"):
                inj_type = observation.insights["injection_type"]
                self._log(f"Detected injection: {inj_type}", "SUCCESS")
                
                analysis = self.planner.analyze_injection(observation.raw_output)
                if analysis.get("is_slow"):
                    self._log("Slow injection - will use Smart Search", "AI")
                
                self.phase = AgentPhase.DATABASE_DISCOVERY
                self._phase_header("PHASE 2: DATABASE DISCOVERY")
        
        elif self.phase == AgentPhase.DATABASE_DISCOVERY:
            if self.memory.current_database:
                self._log(f"Database selected: {self.memory.current_database}", "SUCCESS")
                self.phase = AgentPhase.TABLE_DISCOVERY
                self._phase_header("PHASE 3: TABLE DISCOVERY")
        
        elif self.phase == AgentPhase.TABLE_DISCOVERY:
            if self.memory.tables:
                self._log(f"Found {len(self.memory.tables)} tables", "SUCCESS")
                self.phase = AgentPhase.TABLE_PRIORITIZATION
                self._phase_header("PHASE 4: AI TABLE PRIORITIZATION")
                
                prioritized = self.planner.prioritize_tables(
                    tables=self.memory.tables,
                    categories=self.categories,
                    cms_detected=self.memory.cms_detected,
                    context=self.memory.get_context_for_ai()
                )
                self.pending_tables = prioritized
                
                if prioritized:
                    self._log(f"AI prioritized {len(prioritized)} tables for extraction", "AI")
                    self.phase = AgentPhase.EXTRACTION
                    self._phase_header("PHASE 5: AI-GUIDED EXTRACTION")
        
        elif self.phase == AgentPhase.TABLE_PRIORITIZATION:
            self.phase = AgentPhase.EXTRACTION
            self._phase_header("PHASE 5: AI-GUIDED EXTRACTION")
        
        elif self.phase == AgentPhase.EXTRACTION:
            if not self.pending_tables and self.memory.stats["rows_extracted"] > 0:
                self.phase = AgentPhase.COMPLETION
    
    def _finalize(self) -> Dict:
        """Finalize extraction and return summary."""
        duration = time.time() - self.start_time
        self.memory.stats["duration"] = duration
        
        planner_stats = self.planner.get_stats()
        self.memory.stats["ai_calls"] = planner_stats["ai_calls"]
        self.memory.stats["ai_tokens"] = planner_stats["total_tokens"]
        
        self._phase_header("EXTRACTION COMPLETE")
        
        summary = self.memory.get_summary()
        self._log(f"Duration: {summary['duration']:.1f}s")
        self._log(f"Rounds: {self.round_num}")
        self._log(f"AI calls: {summary['ai_calls']}")
        self._log(f"Rows extracted: {summary['rows_extracted']}")
        
        for cat, count in summary["data_by_category"].items():
            if count > 0:
                self._log(f"  {cat}: {count} records", "SUCCESS")
        
        return summary
