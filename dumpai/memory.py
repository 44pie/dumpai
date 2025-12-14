"""Memory management for DumpAI Agent."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import os


@dataclass
class Action:
    """Single action in history."""
    timestamp: str
    tool: str
    params: Dict
    result: Any
    success: bool
    execution_time: float


@dataclass
class Hypothesis:
    """AI hypothesis about the target."""
    type: str  # cms, injection, valuable_data
    value: str
    confidence: float
    source: str
    timestamp: str = ""


@dataclass 
class Memory:
    """
    Agent memory - stores history, state, and AI context.
    
    Enhanced for AI-driven decision making:
    - hypotheses: AI-generated beliefs about target
    - issue_log: Problems encountered with resolution attempts
    - table_scores: Dynamic utility scores for prioritization
    - strategy_history: Track strategy changes
    """
    
    session_id: str = ""
    start_time: str = ""
    
    databases: List[str] = field(default_factory=list)
    current_database: str = ""
    tables: List[str] = field(default_factory=list)
    columns_cache: Dict[str, List[str]] = field(default_factory=dict)
    
    cms_detected: str = ""
    database_type: str = ""
    injection_type: str = ""
    
    hypotheses: List[Hypothesis] = field(default_factory=list)
    issue_log: List[Dict] = field(default_factory=list)
    table_scores: Dict[str, float] = field(default_factory=dict)
    strategy_history: List[Dict] = field(default_factory=list)
    
    extractions: List[Dict] = field(default_factory=list)
    extracted_data: Dict[str, List[Dict]] = field(default_factory=dict)
    
    history: List[Action] = field(default_factory=list)
    errors: List[Dict] = field(default_factory=list)
    
    stats: Dict = field(default_factory=lambda: {
        "ai_calls": 0,
        "ai_tokens": 0,
        "commands_run": 0,
        "tables_processed": 0,
        "rows_extracted": 0,
        "retries": 0,
        "strategy_changes": 0
    })
    
    def __post_init__(self):
        if not self.session_id:
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not self.start_time:
            self.start_time = datetime.now().isoformat()
        if not self.extracted_data:
            self.extracted_data = {
                "user_data": [],
                "customer_data": [],
                "email_pass": [],
                "api_key": [],
                "sys_data": []
            }
    
    def add_action(self, tool: str, params: Dict, result: Any, 
                   success: bool, execution_time: float):
        """Add action to history."""
        action = Action(
            timestamp=datetime.now().isoformat(),
            tool=tool,
            params=params,
            result=result,
            success=success,
            execution_time=execution_time
        )
        self.history.append(action)
        self.stats["commands_run"] += 1
    
    def add_error(self, tool: str, error: str, context: Dict = None):
        """Add error to log."""
        self.errors.append({
            "timestamp": datetime.now().isoformat(),
            "tool": tool,
            "error": error,
            "context": context or {}
        })
    
    def add_hypothesis(self, type: str, value: str, confidence: float, source: str):
        """Add AI hypothesis about target."""
        h = Hypothesis(
            type=type,
            value=value,
            confidence=confidence,
            source=source,
            timestamp=datetime.now().isoformat()
        )
        self.hypotheses.append(h)
    
    def add_issue(self, issue_type: str, description: str, 
                  resolution: str = "", resolved: bool = False):
        """Log an issue with optional resolution."""
        self.issue_log.append({
            "timestamp": datetime.now().isoformat(),
            "type": issue_type,
            "description": description,
            "resolution": resolution,
            "resolved": resolved
        })
        self.stats["retries"] += 1
    
    def update_table_score(self, table: str, score: float, reason: str = ""):
        """Update utility score for a table."""
        self.table_scores[table] = score
    
    def log_strategy_change(self, old_strategy: str, new_strategy: str, reason: str):
        """Log a strategy change."""
        self.strategy_history.append({
            "timestamp": datetime.now().isoformat(),
            "from": old_strategy,
            "to": new_strategy,
            "reason": reason
        })
        self.stats["strategy_changes"] += 1
    
    def add_extracted_data(self, category: str, rows: List[Dict], 
                           source_table: str = ""):
        """Add extracted data."""
        for row in rows:
            row["_source_table"] = source_table
            row["_extracted_at"] = datetime.now().isoformat()
        
        if category in self.extracted_data:
            self.extracted_data[category].extend(rows)
            self.stats["rows_extracted"] += len(rows)
    
    def get_context_for_ai(self, max_items: int = 10) -> str:
        """Get compressed context for AI prompts."""
        context_parts = []
        
        context_parts.append(f"Database: {self.current_database}")
        context_parts.append(f"CMS: {self.cms_detected or 'Unknown'}")
        context_parts.append(f"Injection: {self.injection_type or 'Unknown'}")
        context_parts.append(f"Tables found: {len(self.tables)}")
        
        if self.tables:
            context_parts.append(f"Sample tables: {', '.join(self.tables[:10])}")
        
        if self.hypotheses:
            context_parts.append("\nAI Hypotheses:")
            for h in self.hypotheses[-5:]:
                context_parts.append(f"  - {h.type}: {h.value} (conf: {h.confidence:.2f})")
        
        if self.table_scores:
            top_tables = sorted(self.table_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            context_parts.append("\nTop priority tables:")
            for table, score in top_tables:
                context_parts.append(f"  - {table}: {score:.2f}")
        
        recent_actions = self.history[-max_items:] if self.history else []
        if recent_actions:
            context_parts.append("\nRecent actions:")
            for action in recent_actions:
                status = "OK" if action.success else "FAIL"
                context_parts.append(f"  - {action.tool}: {status}")
        
        if self.errors:
            recent_errors = self.errors[-3:]
            context_parts.append("\nRecent errors:")
            for err in recent_errors:
                context_parts.append(f"  - {err['tool']}: {err['error'][:50]}")
        
        if self.issue_log:
            unresolved = [i for i in self.issue_log if not i.get('resolved')]
            if unresolved:
                context_parts.append(f"\nUnresolved issues: {len(unresolved)}")
        
        if self.strategy_history:
            last = self.strategy_history[-1]
            context_parts.append(f"\nLast strategy change: {last['from']} -> {last['to']}")
        
        extracted_summary = {k: len(v) for k, v in self.extracted_data.items() if v}
        if extracted_summary:
            context_parts.append(f"\nExtracted: {extracted_summary}")
        
        return "\n".join(context_parts)
    
    def save(self, filepath: str):
        """Save memory to file."""
        data = {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "databases": self.databases,
            "current_database": self.current_database,
            "tables": self.tables,
            "columns_cache": self.columns_cache,
            "cms_detected": self.cms_detected,
            "database_type": self.database_type,
            "extractions": self.extractions,
            "extracted_data": self.extracted_data,
            "history": [
                {
                    "timestamp": a.timestamp,
                    "tool": a.tool,
                    "params": a.params,
                    "success": a.success,
                    "execution_time": a.execution_time
                } for a in self.history
            ],
            "errors": self.errors,
            "stats": self.stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    @classmethod
    def load(cls, filepath: str) -> 'Memory':
        """Load memory from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        memory = cls(
            session_id=data.get("session_id", ""),
            start_time=data.get("start_time", ""),
            databases=data.get("databases", []),
            current_database=data.get("current_database", ""),
            tables=data.get("tables", []),
            columns_cache=data.get("columns_cache", {}),
            cms_detected=data.get("cms_detected", ""),
            database_type=data.get("database_type", ""),
            extractions=data.get("extractions", []),
            extracted_data=data.get("extracted_data", {}),
            errors=data.get("errors", []),
            stats=data.get("stats", {})
        )
        
        for h in data.get("history", []):
            memory.history.append(Action(
                timestamp=h["timestamp"],
                tool=h["tool"],
                params=h["params"],
                result=None,
                success=h["success"],
                execution_time=h["execution_time"]
            ))
        
        return memory
    
    def get_summary(self) -> Dict:
        """Get session summary."""
        return {
            "session_id": self.session_id,
            "duration": self.stats.get("duration", 0),
            "database": self.current_database,
            "cms": self.cms_detected,
            "tables_processed": self.stats["tables_processed"],
            "rows_extracted": self.stats["rows_extracted"],
            "ai_calls": self.stats["ai_calls"],
            "commands_run": self.stats["commands_run"],
            "errors": len(self.errors),
            "data_by_category": {k: len(v) for k, v in self.extracted_data.items()}
        }
