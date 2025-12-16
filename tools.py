"""
DumpAI Tools - SQLMap Execution Wrappers with JSON Parsing & Parallelism

Each tool wraps a specific SQLMap operation with:
- JSON output directory parsing (reliable vs text scraping)
- Parallel execution support
- Thread-safe result aggregation
"""
import csv
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SQLMAP_PATH = os.environ.get("SQLMAP_PATH", "/home/runner/workspace/sqlmap/sqlmap.py")


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    data: Any = None
    raw_output: str = ""
    error: str = ""
    execution_time: float = 0.0
    json_data: Dict = field(default_factory=dict)


class SQLMapOutputParser:
    """Parse SQLMap JSON/CSV output from --output-dir."""
    
    @staticmethod
    def parse_output_dir(output_dir: str, target: str = "") -> Dict:
        """
        Parse SQLMap output directory structure.
        
        Structure:
        output_dir/
          target_host/
            dump/
              database/
                table.csv
            log
            session.sqlite
        """
        result = {
            "databases": [],
            "tables": {},
            "dump_data": {},
            "log": ""
        }
        
        if not os.path.exists(output_dir):
            return result
        
        for host_dir in Path(output_dir).iterdir():
            if not host_dir.is_dir():
                continue
            
            log_file = host_dir / "log"
            if log_file.exists():
                result["log"] = log_file.read_text()
            
            dump_dir = host_dir / "dump"
            if dump_dir.exists():
                for db_dir in dump_dir.iterdir():
                    if db_dir.is_dir():
                        db_name = db_dir.name
                        if db_name not in result["databases"]:
                            result["databases"].append(db_name)
                        
                        result["tables"][db_name] = []
                        
                        for csv_file in db_dir.glob("*.csv"):
                            table_name = csv_file.stem
                            result["tables"][db_name].append(table_name)
                            
                            rows = SQLMapOutputParser.parse_csv(str(csv_file))
                            if rows:
                                key = f"{db_name}.{table_name}"
                                result["dump_data"][key] = rows
        
        return result
    
    @staticmethod
    def parse_csv(csv_path: str) -> List[Dict]:
        """Parse SQLMap CSV dump file."""
        rows = []
        try:
            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(dict(row))
        except Exception:
            pass
        return rows
    
    @staticmethod
    def parse_log_for_databases(log_content: str) -> List[str]:
        """Extract database names from SQLMap log."""
        databases = []
        in_dbs = False
        
        for line in log_content.split("\n"):
            if "available databases" in line.lower():
                in_dbs = True
                continue
            if in_dbs:
                line = line.strip()
                if line.startswith("[*]"):
                    db = line.replace("[*]", "").strip()
                    if db and db not in databases:
                        databases.append(db)
                elif line and not line.startswith("["):
                    in_dbs = False
        
        return databases
    
    @staticmethod
    def parse_log_for_tables(log_content: str) -> List[str]:
        """Extract table names from SQLMap log."""
        tables = []
        
        for line in log_content.split("\n"):
            line = line.strip()
            if line.startswith("| ") and " |" in line:
                table = line.strip("| ").strip()
                if table and not table.startswith("-") and table.lower() != "tables":
                    if table not in tables:
                        tables.append(table)
        
        return tables
    
    @staticmethod
    def parse_log_for_columns(log_content: str) -> List[str]:
        """Extract column names from SQLMap log."""
        columns = []
        
        for line in log_content.split("\n"):
            line = line.strip()
            if line.startswith("| ") and " |" in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if parts and not parts[0].startswith("-"):
                    col = parts[0]
                    if col.lower() not in ["column", "columns", "type"]:
                        if col not in columns:
                            columns.append(col)
        
        return columns
    
    @staticmethod
    def parse_log_for_search_results(log_content: str, pattern: str = "") -> List[str]:
        """Extract search results from SQLMap log.
        
        Parses formats like:
        - [INFO] found table 'database.table'
        - Database: xxx [N tables] | table_name |
        - [INFO] fetching tables like 'pattern' for database 'db'
        """
        tables = []
        current_db = None
        in_table_section = False
        
        for line in log_content.split("\n"):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            
            # Parse "Database: xxx"
            if line_stripped.startswith("Database:"):
                current_db = line_stripped.split(":", 1)[1].strip()
                continue
            
            # Parse "[N table(s)]"
            if re.match(r'\[\d+ tables?\]', line_stripped):
                in_table_section = True
                continue
            
            # Parse "| table_name |" format
            if line_stripped.startswith("|") and line_stripped.endswith("|"):
                table_name = line_stripped.strip("| ").strip()
                if table_name and not table_name.startswith("-") and table_name.lower() not in ["tables", "table"]:
                    if len(table_name) > 2:
                        if current_db:
                            table_ref = f"{current_db}.{table_name}"
                        else:
                            table_ref = table_name
                        if table_ref not in tables:
                            tables.append(table_ref)
                            in_table_section = True
            
            # End of table section
            if line_stripped.startswith("+") and "---" in line_stripped and in_table_section:
                in_table_section = False
            
            # Parse "[INFO] found table 'db.table'"
            if "found table" in line_lower or "retrieved table" in line_lower:
                match = re.search(r"'([^']+)'", line)
                if match:
                    table_ref = match.group(1)
                    if table_ref not in tables and len(table_ref) > 2:
                        tables.append(table_ref)
            
            # Parse "fetching tables like 'pattern' for database 'db'"
            if "fetching tables like" in line_lower:
                match = re.search(r"for database\s+'([^']+)'", line, re.IGNORECASE)
                if match:
                    current_db = match.group(1)
        
        return tables
    
    @staticmethod
    def read_log_from_output_dir(output_dir: str) -> str:
        """Read SQLMap log file from output directory."""
        if not os.path.exists(output_dir):
            return ""
        
        # SQLMap saves logs in output_dir/target_host/log
        for host_dir in Path(output_dir).iterdir():
            if host_dir.is_dir():
                log_file = host_dir / "log"
                if log_file.exists():
                    return log_file.read_text(encoding='utf-8', errors='ignore')
        
        return ""


class BaseTool:
    """Base class for all tools with JSON output support."""
    
    def __init__(self, config: Dict, verbose: bool = False, timeout: int = 300,
                 output_base: str = ""):
        self.config = config
        self.verbose = verbose
        self.timeout = timeout
        self.base_cmd = config.get("base_cmd", "")
        self.output_base = output_base or tempfile.mkdtemp(prefix="sqlmap_")
        self._lock = threading.Lock()
        self._user_output_dir = self._extract_user_output_dir()
    
    def _extract_user_output_dir(self) -> str:
        """Extract user's --output-dir from base command if present."""
        import re
        match = re.search(r'--output-dir[=\s]+["\']?([^"\'\s]+)["\']?', self.base_cmd)
        return match.group(1) if match else ""
    
    def _get_output_dir(self) -> str:
        """Get unique output directory for this run."""
        run_id = str(uuid.uuid4())[:8]
        return os.path.join(self.output_base, f"run_{run_id}")
    
    def _build_cmd(self, extra_args: List[str], output_dir: str = None) -> str:
        """Build SQLMap command with extra arguments.
        
        IMPORTANT: User's --output-dir is PRESERVED to use SQLMap's cached session.
        Only adds temp output-dir if user didn't specify one.
        """
        import re
        cmd = self.base_cmd
        
        if "sqlmap" in cmd and SQLMAP_PATH and os.path.exists(SQLMAP_PATH):
            match = re.search(r'(.*?)(python3\s+\S*sqlmap\.py|sqlmap)\s', cmd)
            if match:
                prefix = match.group(1)
                sqlmap_part = cmd[match.end()-1:]
                cmd = f"{prefix}python3 {SQLMAP_PATH} {sqlmap_part}"
            elif cmd.startswith("sqlmap "):
                cmd = f"python3 {SQLMAP_PATH} " + cmd[7:]
            else:
                cmd = cmd.replace("sqlmap ", f"python3 {SQLMAP_PATH} ")
        
        if "--batch" not in cmd:
            cmd += " --batch"
        
        if "--ignore-stdin" not in cmd:
            cmd += " --ignore-stdin"
        
        # CRITICAL: Answer YES to important prompts (common column check, etc.)
        # --batch alone answers N to many questions which breaks extraction
        if "--answers" not in cmd:
            cmd += ' --answers="Y"'
        
        # Force single thread for time-based techniques to avoid multithreading warning
        # Check if technique contains T (time-based) - multithreading is unsafe
        technique_match = re.search(r'--technique[=\s]+([A-Z]+)', cmd, re.IGNORECASE)
        if technique_match:
            technique = technique_match.group(1).upper()
            if 'T' in technique and "--threads" in cmd:
                # Force threads=1 for time-based to avoid the warning/prompt
                cmd = re.sub(r'--threads[=\s]+\d+', '--threads=1', cmd)
        
        # Only add temp output-dir if user didn't specify one
        if "--output-dir" not in cmd:
            temp_dir = tempfile.mkdtemp(prefix="sqlmap_")
            cmd += f" --output-dir={temp_dir}"
        
        # Optimize --technique: use only available techniques in optimal order
        available = self.config.get("available_techniques", "")
        technique_match = re.search(r'--technique[=\s]+([A-Z]+)', cmd, re.IGNORECASE)
        
        if technique_match:
            original = technique_match.group(1).upper()
            
            if available:
                # Filter to only available techniques and order by speed
                priority = {'U': 0, 'E': 1, 'S': 2, 'Q': 3, 'B': 4, 'T': 5}
                # Keep only techniques that are both requested AND available
                filtered = [t for t in original if t in available]
                optimized = ''.join(sorted(filtered, key=lambda x: priority.get(x, 99)))
            else:
                # No available info yet, just reorder by speed
                priority = {'U': 0, 'E': 1, 'S': 2, 'Q': 3, 'B': 4, 'T': 5}
                optimized = ''.join(sorted(original, key=lambda x: priority.get(x, 99)))
            
            if optimized and optimized != original:
                cmd = re.sub(r'--technique[=\s]+[A-Z]+', f'--technique={optimized}', cmd, flags=re.IGNORECASE)
        
        for arg in extra_args:
            cmd += f" {arg}"
        
        # Apply extra_flags from adaptive retry (WAF bypass, timeouts, etc.)
        extra_flags = self.config.get("extra_flags", "")
        if extra_flags:
            cmd += f" {extra_flags}"
        
        # Apply threads override (e.g., for time-based techniques)
        threads_override = self.config.get("threads_override")
        if threads_override:
            import re
            if "--threads" in cmd:
                cmd = re.sub(r'--threads[=\s]+\d+', f'--threads={threads_override}', cmd)
            else:
                cmd += f" --threads={threads_override}"
        
        return cmd
    
    def _run_cmd(self, cmd: str, stream_output: bool = False) -> Tuple[str, str, int]:
        """Run command and return (stdout, stderr, returncode)."""
        if self.verbose:
            print(f"[*] CMD: {cmd}")
        
        try:
            if stream_output and self.verbose:
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                output_lines = []
                for line in iter(process.stdout.readline, ''):
                    if line:
                        print(line, end='', flush=True)
                        output_lines.append(line)
                
                process.wait(timeout=self.timeout)
                output = ''.join(output_lines)
                return output, "", process.returncode
            else:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", "Timeout expired", -1
        except Exception as e:
            return "", str(e), -1
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool. Override in subclasses."""
        raise NotImplementedError


class EnumerateDBs(BaseTool):
    """Enumerate available databases."""
    
    def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        output_dir = self._get_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = self._build_cmd(["--dbs"], output_dir)
        stdout, stderr, code = self._run_cmd(cmd, stream_output=self.verbose)
        
        output = stdout + stderr
        
        parsed = SQLMapOutputParser.parse_output_dir(output_dir)
        databases = parsed.get("databases", [])
        
        if not databases:
            databases = SQLMapOutputParser.parse_log_for_databases(output)
        
        return ToolResult(
            success=len(databases) > 0 or "vulnerable" in output.lower(),
            data=databases,
            raw_output=output,
            error="" if databases else "No databases found",
            execution_time=time.time() - start,
            json_data=parsed
        )


class EnumerateTables(BaseTool):
    """Enumerate tables in a database."""
    
    def execute(self, database: str = "", **kwargs) -> ToolResult:
        start = time.time()
        output_dir = self._get_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        
        args = ["--tables"]
        if database:
            args.append(f"-D {database}")
        
        cmd = self._build_cmd(args, output_dir)
        stdout, stderr, code = self._run_cmd(cmd, stream_output=self.verbose)
        
        output = stdout + stderr
        
        parsed = SQLMapOutputParser.parse_output_dir(output_dir)
        tables = parsed.get("tables", {}).get(database, [])
        
        if not tables:
            tables = SQLMapOutputParser.parse_log_for_tables(output)
        
        return ToolResult(
            success=len(tables) > 0,
            data=tables,
            raw_output=output,
            error="" if tables else "No tables found",
            execution_time=time.time() - start,
            json_data=parsed
        )


class GetColumns(BaseTool):
    """Get columns for a specific table."""
    
    def execute(self, database: str = "", table: str = "", **kwargs) -> ToolResult:
        start = time.time()
        output_dir = self._get_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        
        args = ["--columns"]
        
        if database:
            args.append(f"-D {database}")
        if table:
            args.append(f"-T {table}")
        
        cmd = self._build_cmd(args, output_dir)
        stdout, stderr, code = self._run_cmd(cmd, stream_output=self.verbose)
        
        output = stdout + stderr
        columns = SQLMapOutputParser.parse_log_for_columns(output)
        
        return ToolResult(
            success=len(columns) > 0,
            data=columns,
            raw_output=output,
            error="" if columns else "No columns found",
            execution_time=time.time() - start
        )


class SearchTables(BaseTool):
    """Search tables by pattern - PARALLEL execution."""
    
    def execute(self, patterns: List[str] = None, database: str = "", parallel: bool = True, 
                max_workers: int = 5, **kwargs) -> ToolResult:
        start = time.time()
        
        if not patterns:
            patterns = ["user", "admin", "employee", "customer", "config"]
        
        all_tables = []
        all_output = []
        target_db = database
        
        IGNORE_TLDS = {'com', 'org', 'net', 'io', 'edu', 'gov', 'co', 'ru', 'de', 'uk', 'fr', 'es', 'it', 'nl', 'be', 'ch', 'at', 'pl', 'cz', 'sk', 'hu', 'ro', 'bg', 'ua', 'by', 'kz', 'cn', 'jp', 'kr', 'in', 'br', 'mx', 'ar', 'cl', 'au', 'nz', 'za', 'eg', 'ng', 'ke', 'info', 'biz', 'tv', 'me', 'cc', 'ws', 'us', 'ca', 'eu'}
        
        def search_pattern(pattern: str) -> Tuple[str, str, List[str]]:
            output_dir = self._get_output_dir()
            os.makedirs(output_dir, exist_ok=True)
            
            args = [f"--search -T {pattern}"]
            if target_db:
                args.append(f"-D {target_db}")
            cmd = self._build_cmd(args, output_dir)
            stdout, stderr, code = self._run_cmd(cmd, stream_output=False)
            output = stdout + stderr
            
            tables = []
            current_db = target_db or None
            in_table_section = False
            
            for line in output.split("\n"):
                line_stripped = line.strip()
                line_lower = line_stripped.lower()
                
                # Parse "Database: xxx"
                if line_stripped.startswith("Database:"):
                    current_db = line_stripped.split(":", 1)[1].strip()
                    continue
                
                # Parse "[N table(s)]" marker
                if re.match(r'\[\d+ tables?\]', line_stripped):
                    in_table_section = True
                    continue
                
                # Parse "| table_name |" format
                if in_table_section and line_stripped.startswith("|") and line_stripped.endswith("|"):
                    table_name = line_stripped.strip("| ").strip()
                    if table_name and not table_name.startswith("-") and table_name.lower() != "tables":
                        if table_name.lower() not in IGNORE_TLDS and len(table_name) > 2:
                            if current_db:
                                table_ref = f"{current_db}.{table_name}"
                            else:
                                table_ref = table_name
                            if table_ref not in tables:
                                tables.append(table_ref)
                
                # End of table section
                if line_stripped.startswith("+") and "-" in line_stripped and in_table_section:
                    if tables:
                        in_table_section = False
                
                # Parse "[INFO] found table 'db.table'" format
                if "found" in line_lower and "table" in line_lower:
                    match = re.search(r"'([^']+)'", line)
                    if match:
                        table_ref = match.group(1)
                        parts = table_ref.split('.')
                        if len(parts) == 2:
                            db, tbl = parts
                            if tbl.lower() not in IGNORE_TLDS and len(tbl) > 2:
                                if table_ref not in tables:
                                    tables.append(table_ref)
                        elif len(parts) == 1 and len(table_ref) > 2:
                            if table_ref not in tables:
                                tables.append(table_ref)
                
                # Parse "[INFO] searching table 'table'" - may indicate match
                if "searching table" in line_lower or "found table" in line_lower:
                    match = re.search(r"'([^']+)'", line)
                    if match:
                        tbl = match.group(1)
                        if '.' in tbl:
                            if tbl not in tables:
                                tables.append(tbl)
                        elif current_db and len(tbl) > 2:
                            table_ref = f"{current_db}.{tbl}"
                            if table_ref not in tables:
                                tables.append(table_ref)
                
                # Parse "like '...%pattern%...'" results
                if "like" in line_lower and pattern.lower() in line_lower:
                    match = re.search(r"table[^']*'([^']+)'", line, re.IGNORECASE)
                    if match:
                        tbl = match.group(1).replace('%', '')
                        if tbl and len(tbl) > 2 and tbl.lower() not in IGNORE_TLDS:
                            if current_db:
                                table_ref = f"{current_db}.{tbl}"
                            else:
                                table_ref = tbl
                            if table_ref not in tables:
                                tables.append(table_ref)
            
            # Fallback 1: Parse log file from output directory for search results
            # Check both our temp dir AND user's --output-dir (SQLMap writes to user's dir)
            dirs_to_check = [output_dir]
            if self._user_output_dir:
                dirs_to_check.append(self._user_output_dir)
            
            for check_dir in dirs_to_check:
                log_content = SQLMapOutputParser.read_log_from_output_dir(check_dir)
                if log_content:
                    log_tables = SQLMapOutputParser.parse_log_for_search_results(log_content, pattern)
                    for tbl in log_tables:
                        if tbl not in tables:
                            tables.append(tbl)
            
            # Fallback 2: parse output directory for any saved tables (dump CSVs)
            parsed = SQLMapOutputParser.parse_output_dir(output_dir)
            if parsed.get("tables"):
                for db, db_tables in parsed["tables"].items():
                    for tbl in db_tables:
                        if tbl.lower() not in IGNORE_TLDS and len(tbl) > 2:
                            table_ref = f"{db}.{tbl}"
                            if table_ref not in tables:
                                tables.append(table_ref)
            
            # Debug: print results for this pattern
            if self.verbose:
                if tables:
                    print(f"[SEARCH] Pattern '{pattern}' found: {tables}")
                else:
                    # Save raw output for debugging when no tables found
                    debug_file = os.path.join(output_dir, f"search_debug_{pattern}.txt")
                    with open(debug_file, 'w') as f:
                        f.write(f"Pattern: {pattern}\nDatabase: {target_db}\n\n")
                        f.write(output)
                    print(f"[SEARCH] Pattern '{pattern}' found nothing. Debug: {debug_file}")
            
            return pattern, output, tables
        
        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(search_pattern, p): p for p in patterns}
                for future in as_completed(futures):
                    pattern, output, tables = future.result()
                    all_output.append(output)
                    for t in tables:
                        if t not in all_tables:
                            all_tables.append(t)
        else:
            for pattern in patterns:
                _, output, tables = search_pattern(pattern)
                all_output.append(output)
                for t in tables:
                    if t not in all_tables:
                        all_tables.append(t)
        
        return ToolResult(
            success=len(all_tables) > 0,
            data=all_tables,
            raw_output="\n".join(all_output),
            error="" if all_tables else "No tables found matching patterns",
            execution_time=time.time() - start
        )


class SearchColumns(BaseTool):
    """Search columns by pattern - PARALLEL execution."""
    
    def execute(self, patterns: List[str] = None, parallel: bool = True,
                max_workers: int = 5, **kwargs) -> ToolResult:
        start = time.time()
        
        if not patterns:
            patterns = ["password", "email", "api_key", "token"]
        
        all_columns = []
        all_output = []
        
        def search_pattern(pattern: str) -> Tuple[str, str, List[str]]:
            output_dir = self._get_output_dir()
            os.makedirs(output_dir, exist_ok=True)
            
            args = [f"--search -C {pattern}"]
            cmd = self._build_cmd(args, output_dir)
            stdout, stderr, code = self._run_cmd(cmd, stream_output=False)
            output = stdout + stderr
            
            columns = []
            for line in output.split("\n"):
                if "found" in line.lower():
                    match = re.search(r'(\w+\.\w+\.\w+)', line)
                    if match:
                        columns.append(match.group(1))
            
            return pattern, output, columns
        
        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(search_pattern, p): p for p in patterns}
                for future in as_completed(futures):
                    pattern, output, columns = future.result()
                    all_output.append(output)
                    for c in columns:
                        if c not in all_columns:
                            all_columns.append(c)
        else:
            for pattern in patterns:
                _, output, columns = search_pattern(pattern)
                all_output.append(output)
                for c in columns:
                    if c not in all_columns:
                        all_columns.append(c)
        
        return ToolResult(
            success=len(all_columns) > 0,
            data=all_columns,
            raw_output="\n".join(all_output),
            error="" if all_columns else "No columns found",
            execution_time=time.time() - start
        )


class DumpTable(BaseTool):
    """Dump entire table with JSON/CSV parsing."""
    
    def execute(self, database: str = "", table: str = "", 
                max_rows: int = 0, **kwargs) -> ToolResult:
        start = time.time()
        output_dir = self._get_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        
        args = ["--dump"]
        if database:
            args.append(f"-D {database}")
        if table:
            args.append(f"-T {table}")
        if max_rows > 0:
            args.append(f"--stop {max_rows}")
        
        cmd = self._build_cmd(args, output_dir)
        stdout, stderr, code = self._run_cmd(cmd, stream_output=self.verbose)
        
        output = stdout + stderr
        
        parsed = SQLMapOutputParser.parse_output_dir(output_dir)
        key = f"{database}.{table}" if database and table else ""
        rows = parsed.get("dump_data", {}).get(key, [])
        
        if not rows:
            rows = self._parse_table_output(output)
        
        return ToolResult(
            success=len(rows) > 0,
            data=rows,
            raw_output=output,
            error="" if rows else "No data extracted",
            execution_time=time.time() - start,
            json_data=parsed
        )
    
    def _parse_table_output(self, output: str) -> List[Dict]:
        """Fallback: Parse SQLMap text table output."""
        rows = []
        headers = []
        
        lines = output.split("\n")
        for line in lines:
            line = line.strip()
            
            if line.startswith("| ") and " |" in line and "---" not in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                
                if not headers:
                    headers = parts
                else:
                    if len(parts) == len(headers):
                        row = dict(zip(headers, parts))
                        rows.append(row)
        
        return rows


class DumpColumns(BaseTool):
    """Dump specific columns - supports PARALLEL per-column execution."""
    
    def execute(self, database: str = "", table: str = "", 
                columns: List[str] = None, max_rows: int = 0,
                parallel: bool = True, max_workers: int = 5, **kwargs) -> ToolResult:
        start = time.time()
        
        if not columns:
            return ToolResult(
                success=False,
                error="No columns specified",
                execution_time=time.time() - start
            )
        
        if parallel and len(columns) > 1:
            return self._parallel_dump(database, table, columns, max_rows, max_workers)
        else:
            return self._single_dump(database, table, columns, max_rows)
    
    def _single_dump(self, database: str, table: str, columns: List[str], 
                     max_rows: int) -> ToolResult:
        """Dump all columns in single command."""
        start = time.time()
        
        args = ["--dump"]
        if database:
            args.append(f"-D {database}")
        if table:
            args.append(f"-T {table}")
        if columns:
            cols_str = ",".join(columns)
            args.append(f"-C {cols_str}")
        if max_rows > 0:
            args.append(f"--stop {max_rows}")
        
        cmd = self._build_cmd(args)
        stdout, stderr, code = self._run_cmd(cmd, stream_output=self.verbose)
        
        output = stdout + stderr
        
        parsed = {}
        if self._user_output_dir:
            parsed = SQLMapOutputParser.parse_output_dir(self._user_output_dir)
        key = f"{database}.{table}" if database and table else ""
        rows = parsed.get("dump_data", {}).get(key, [])
        
        if not rows:
            rows = self._parse_table_output(output)
        
        return ToolResult(
            success=len(rows) > 0,
            data=rows,
            raw_output=output,
            error="" if rows else "No data extracted",
            execution_time=time.time() - start,
            json_data=parsed
        )
    
    def _parallel_dump(self, database: str, table: str, columns: List[str],
                       max_rows: int, max_workers: int) -> ToolResult:
        """Dump each column in parallel with isolated output dirs, then merge results."""
        start = time.time()
        all_output = []
        column_data = {}
        errors = []
        
        def dump_single_column(col: str) -> Tuple[str, List[Dict], str, str]:
            # CRITICAL: Each parallel column worker needs ISOLATED output dir
            # Otherwise race condition: multiple workers overwrite same CSV!
            isolated_dir = tempfile.mkdtemp(prefix=f"sqlmap_col_{col[:10]}_")
            
            args = ["--dump", f"--output-dir={isolated_dir}"]
            if database:
                args.append(f"-D {database}")
            if table:
                args.append(f"-T {table}")
            args.append(f"-C {col}")
            if max_rows > 0:
                args.append(f"--stop {max_rows}")
            
            # Build command WITHOUT user's output-dir to avoid race condition
            cmd = self.base_cmd
            if "--batch" not in cmd:
                cmd += " --batch"
            if "--ignore-stdin" not in cmd:
                cmd += " --ignore-stdin"
            if "--answers" not in cmd:
                cmd += ' --answers="Y"'
            for arg in args:
                cmd += f" {arg}"
            
            stdout, stderr, code = self._run_cmd(cmd, stream_output=False)
            output = stdout + stderr
            
            # Parse from isolated directory
            parsed = SQLMapOutputParser.parse_output_dir(isolated_dir)
            key = f"{database}.{table}" if database and table else ""
            rows = parsed.get("dump_data", {}).get(key, [])
            
            if not rows:
                rows = self._parse_single_column_output(output, col)
            
            # Cleanup isolated dir
            try:
                import shutil
                shutil.rmtree(isolated_dir, ignore_errors=True)
            except:
                pass
            
            return col, rows, output, "" if rows else f"No data for {col}"
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(dump_single_column, col): col for col in columns}
            
            for future in as_completed(futures):
                col, rows, output, error = future.result()
                all_output.append(output)
                
                if rows:
                    column_data[col] = [row.get(col, "") for row in rows]
                if error:
                    errors.append(error)
        
        merged_rows = self._merge_column_data(column_data)
        
        return ToolResult(
            success=len(merged_rows) > 0,
            data=merged_rows,
            raw_output="\n---\n".join(all_output),
            error="; ".join(errors) if errors and not merged_rows else "",
            execution_time=time.time() - start
        )
    
    def _parse_single_column_output(self, output: str, column: str) -> List[Dict]:
        """Parse output for a single column dump."""
        rows = []
        
        for line in output.split("\n"):
            line = line.strip()
            if line.startswith("| ") and " |" in line and "---" not in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 1 and parts[0].lower() != column.lower():
                    rows.append({column: parts[0]})
        
        return rows
    
    def _merge_column_data(self, column_data: Dict[str, List]) -> List[Dict]:
        """Merge parallel column results into unified rows."""
        if not column_data:
            return []
        
        max_rows = max(len(v) for v in column_data.values())
        merged = []
        
        for i in range(max_rows):
            row = {}
            for col, values in column_data.items():
                row[col] = values[i] if i < len(values) else ""
            merged.append(row)
        
        return merged
    
    def _parse_table_output(self, output: str) -> List[Dict]:
        """Fallback: Parse SQLMap text table output."""
        rows = []
        headers = []
        
        for line in output.split("\n"):
            line = line.strip()
            
            if line.startswith("| ") and " |" in line and "---" not in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                
                if not headers:
                    headers = parts
                else:
                    if len(parts) == len(headers):
                        row = dict(zip(headers, parts))
                        rows.append(row)
        
        return rows


class ParallelTableDumper:
    """Orchestrates parallel dumping of multiple tables with isolated output dirs."""
    
    def __init__(self, config: Dict, verbose: bool = False, 
                 max_table_workers: int = 3, max_column_workers: int = 5,
                 output_base: str = ""):
        self.config = config
        self.verbose = verbose
        self.max_table_workers = max_table_workers
        self.max_column_workers = max_column_workers
        self.output_base = output_base or tempfile.gettempdir()
        self._lock = threading.Lock()
        self._results = {}
    
    def _create_isolated_output_dir(self, prefix: str = "table") -> str:
        """Create isolated output directory for each worker."""
        return tempfile.mkdtemp(prefix=f"sqlmap_{prefix}_")
    
    def dump_tables_parallel(self, database: str, tables: List[str],
                             columns_map: Dict[str, List[str]] = None,
                             max_rows: int = 0) -> Dict[str, ToolResult]:
        """
        Dump multiple tables in parallel with isolated output directories.
        
        Each table gets its own temporary output directory to prevent
        race conditions and data contamination between parallel workers.
        """
        columns_map = columns_map or {}
        results = {}
        
        def dump_table(table: str) -> Tuple[str, ToolResult]:
            isolated_output = self._create_isolated_output_dir(f"table_{table[:10]}")
            cols = columns_map.get(table, [])
            
            try:
                if cols:
                    tool = DumpColumns(
                        config=dict(self.config),
                        verbose=False,
                        output_base=isolated_output
                    )
                    result = tool.execute(
                        database=database,
                        table=table,
                        columns=cols,
                        max_rows=max_rows,
                        parallel=True,
                        max_workers=self.max_column_workers
                    )
                else:
                    tool = DumpTable(
                        config=dict(self.config),
                        verbose=False,
                        output_base=isolated_output
                    )
                    result = tool.execute(
                        database=database,
                        table=table,
                        max_rows=max_rows
                    )
                
                return table, result
            finally:
                pass
        
        if self.verbose:
            print(f"[*] Parallel dump: {len(tables)} tables, {self.max_table_workers} workers")
        
        with ThreadPoolExecutor(max_workers=self.max_table_workers) as executor:
            futures = {executor.submit(dump_table, t): t for t in tables}
            
            for future in as_completed(futures):
                table, result = future.result()
                
                with self._lock:
                    results[table] = result
                    
                    if self.verbose:
                        status = "OK" if result.success else "FAIL"
                        rows = len(result.data) if result.data else 0
                        print(f"  [{status}] {table}: {rows} rows")
        
        return results


class AnalyzeSchema(BaseTool):
    """Analyze database schema."""
    
    def execute(self, database: str = "", **kwargs) -> ToolResult:
        start = time.time()
        output_dir = self._get_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        
        args = ["--schema"]
        if database:
            args.append(f"-D {database}")
        
        cmd = self._build_cmd(args, output_dir)
        stdout, stderr, code = self._run_cmd(cmd, stream_output=self.verbose)
        
        output = stdout + stderr
        
        return ToolResult(
            success="schema" in output.lower() or len(output) > 100,
            data={"schema": output},
            raw_output=output,
            error="" if output else "Failed to get schema",
            execution_time=time.time() - start
        )


class AnalyzeColumns(BaseTool):
    """Analyze column types and patterns."""
    
    def execute(self, database: str = "", table: str = "", **kwargs) -> ToolResult:
        columns_tool = GetColumns(self.config, self.verbose, self.timeout, self.output_base)
        result = columns_tool.execute(database=database, table=table)
        
        if not result.success:
            return result
        
        analysis = {
            "columns": result.data,
            "patterns": self._analyze_patterns(result.data)
        }
        
        return ToolResult(
            success=True,
            data=analysis,
            raw_output=result.raw_output,
            execution_time=result.execution_time
        )
    
    def _analyze_patterns(self, columns: List[str]) -> Dict:
        """Analyze column patterns."""
        patterns = {
            "credential": [],
            "pii": [],
            "system": [],
            "api": []
        }
        
        cred_kw = ["password", "passwd", "pwd", "hash", "salt", "secret"]
        pii_kw = ["email", "phone", "address", "name", "firstname", "lastname"]
        sys_kw = ["config", "setting", "option", "host", "server"]
        api_kw = ["api", "key", "token", "oauth", "credential"]
        
        for col in columns:
            col_lower = col.lower()
            
            if any(kw in col_lower for kw in cred_kw):
                patterns["credential"].append(col)
            if any(kw in col_lower for kw in pii_kw):
                patterns["pii"].append(col)
            if any(kw in col_lower for kw in sys_kw):
                patterns["system"].append(col)
            if any(kw in col_lower for kw in api_kw):
                patterns["api"].append(col)
        
        return patterns
