"""
DumpAI Tools - SQLMap Execution Wrappers

Each tool wraps a specific SQLMap operation with:
- Proper command construction
- Output parsing
- Result formatting for AI analysis
"""
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


SQLMAP_PATH = os.environ.get("SQLMAP_PATH", "/home/runner/workspace/sqlmap/sqlmap.py")


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    data: Any = None
    raw_output: str = ""
    error: str = ""
    execution_time: float = 0.0


class BaseTool:
    """Base class for all tools."""
    
    def __init__(self, config: Dict, verbose: bool = False, timeout: int = 300):
        self.config = config
        self.verbose = verbose
        self.timeout = timeout
        self.base_cmd = config.get("base_cmd", "")
    
    def _build_cmd(self, extra_args: List[str]) -> str:
        """Build SQLMap command with extra arguments."""
        cmd = self.base_cmd
        
        # Replace sqlmap with full path
        if "sqlmap" in cmd and SQLMAP_PATH and os.path.exists(SQLMAP_PATH):
            if cmd.startswith("sqlmap "):
                cmd = f"python3 {SQLMAP_PATH} " + cmd[7:]
            else:
                cmd = cmd.replace("sqlmap ", f"python3 {SQLMAP_PATH} ")
        
        if "--batch" not in cmd:
            cmd += " --batch"
        
        # Add --ignore-stdin for subprocess compatibility
        if "--ignore-stdin" not in cmd:
            cmd += " --ignore-stdin"
        
        # Add extra args
        for arg in extra_args:
            cmd += f" {arg}"
        
        return cmd
    
    def _run_cmd(self, cmd: str) -> tuple:
        """Run command and return (stdout, stderr, returncode)."""
        if self.verbose:
            print(f"[*] CMD: {cmd}")
        
        try:
            if self.verbose:
                # Stream output in real-time for verbose mode
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
                # Capture output silently
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
        cmd = self._build_cmd(["--dbs"])
        stdout, stderr, code = self._run_cmd(cmd)
        
        databases = []
        output = stdout + stderr
        
        in_dbs = False
        for line in output.split("\n"):
            if "available databases" in line.lower():
                in_dbs = True
                continue
            if in_dbs:
                line = line.strip()
                if line.startswith("[*]"):
                    db = line.replace("[*]", "").strip()
                    if db and db not in databases:
                        databases.append(db)
        
        return ToolResult(
            success=len(databases) > 0 or "vulnerable" in output.lower(),
            data=databases,
            raw_output=output,
            error="" if databases else "No databases found",
            execution_time=time.time() - start
        )


class EnumerateTables(BaseTool):
    """Enumerate tables in a database."""
    
    def execute(self, database: str = "", **kwargs) -> ToolResult:
        start = time.time()
        
        args = ["--tables"]
        if database:
            args.append(f"-D {database}")
        
        cmd = self._build_cmd(args)
        stdout, stderr, code = self._run_cmd(cmd)
        
        tables = []
        output = stdout + stderr
        
        for line in output.split("\n"):
            line = line.strip()
            if line.startswith("| ") and " |" in line:
                table = line.strip("| ").strip()
                if table and not table.startswith("-") and table.lower() != "tables":
                    tables.append(table)
        
        return ToolResult(
            success=len(tables) > 0,
            data=tables,
            raw_output=output,
            error="" if tables else "No tables found",
            execution_time=time.time() - start
        )


class GetColumns(BaseTool):
    """Get columns for a specific table."""
    
    def execute(self, database: str = "", table: str = "", **kwargs) -> ToolResult:
        start = time.time()
        
        args = ["--columns"]
        if database:
            args.append(f"-D {database}")
        if table:
            args.append(f"-T {table}")
        
        cmd = self._build_cmd(args)
        stdout, stderr, code = self._run_cmd(cmd)
        
        columns = []
        output = stdout + stderr
        
        for line in output.split("\n"):
            line = line.strip()
            if line.startswith("| ") and " |" in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if parts and not parts[0].startswith("-"):
                    col = parts[0]
                    if col.lower() not in ["column", "columns", "type"]:
                        columns.append(col)
        
        return ToolResult(
            success=len(columns) > 0,
            data=columns,
            raw_output=output,
            error="" if columns else "No columns found",
            execution_time=time.time() - start
        )


class SearchTables(BaseTool):
    """Search tables by pattern."""
    
    def execute(self, patterns: List[str] = None, parallel: bool = True, 
                max_workers: int = 5, **kwargs) -> ToolResult:
        start = time.time()
        
        if not patterns:
            patterns = ["user", "admin", "employee", "customer", "config"]
        
        all_tables = []
        all_output = []
        
        def search_pattern(pattern: str) -> tuple:
            args = [f"--search -T {pattern}"]
            cmd = self._build_cmd(args)
            stdout, stderr, code = self._run_cmd(cmd)
            return pattern, stdout + stderr
        
        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(search_pattern, p): p for p in patterns}
                for future in as_completed(futures):
                    pattern, output = future.result()
                    all_output.append(output)
                    
                    for line in output.split("\n"):
                        if "found" in line.lower() or line.strip().startswith("["):
                            match = re.search(r'(\w+\.\w+)', line)
                            if match:
                                table = match.group(1)
                                if table not in all_tables:
                                    all_tables.append(table)
        else:
            for pattern in patterns:
                _, output = search_pattern(pattern)
                all_output.append(output)
        
        return ToolResult(
            success=len(all_tables) > 0,
            data=all_tables,
            raw_output="\n".join(all_output),
            error="" if all_tables else "No tables found matching patterns",
            execution_time=time.time() - start
        )


class SearchColumns(BaseTool):
    """Search columns by pattern."""
    
    def execute(self, patterns: List[str] = None, **kwargs) -> ToolResult:
        start = time.time()
        
        if not patterns:
            patterns = ["password", "email", "api_key", "token"]
        
        all_columns = []
        all_output = []
        
        for pattern in patterns:
            args = [f"--search -C {pattern}"]
            cmd = self._build_cmd(args)
            stdout, stderr, code = self._run_cmd(cmd)
            output = stdout + stderr
            all_output.append(output)
            
            for line in output.split("\n"):
                if "found" in line.lower():
                    match = re.search(r'(\w+\.\w+\.\w+)', line)
                    if match:
                        col = match.group(1)
                        if col not in all_columns:
                            all_columns.append(col)
        
        return ToolResult(
            success=len(all_columns) > 0,
            data=all_columns,
            raw_output="\n".join(all_output),
            error="" if all_columns else "No columns found",
            execution_time=time.time() - start
        )


class DumpTable(BaseTool):
    """Dump entire table."""
    
    def execute(self, database: str = "", table: str = "", 
                max_rows: int = 0, **kwargs) -> ToolResult:
        start = time.time()
        
        args = ["--dump"]
        if database:
            args.append(f"-D {database}")
        if table:
            args.append(f"-T {table}")
        if max_rows > 0:
            args.append(f"--stop {max_rows}")
        
        cmd = self._build_cmd(args)
        stdout, stderr, code = self._run_cmd(cmd)
        
        output = stdout + stderr
        rows = self._parse_table_output(output)
        
        return ToolResult(
            success=len(rows) > 0,
            data=rows,
            raw_output=output,
            error="" if rows else "No data extracted",
            execution_time=time.time() - start
        )
    
    def _parse_table_output(self, output: str) -> List[Dict]:
        """Parse SQLMap table output into list of dicts."""
        rows = []
        headers = []
        
        lines = output.split("\n")
        for i, line in enumerate(lines):
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
    """Dump specific columns from a table."""
    
    def execute(self, database: str = "", table: str = "", 
                columns: List[str] = None, max_rows: int = 0, **kwargs) -> ToolResult:
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
        stdout, stderr, code = self._run_cmd(cmd)
        
        output = stdout + stderr
        rows = self._parse_table_output(output)
        
        return ToolResult(
            success=len(rows) > 0,
            data=rows,
            raw_output=output,
            error="" if rows else "No data extracted",
            execution_time=time.time() - start
        )
    
    def _parse_table_output(self, output: str) -> List[Dict]:
        """Parse SQLMap table output into list of dicts."""
        rows = []
        headers = []
        
        lines = output.split("\n")
        for i, line in enumerate(lines):
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


class AnalyzeSchema(BaseTool):
    """Analyze database schema."""
    
    def execute(self, database: str = "", **kwargs) -> ToolResult:
        start = time.time()
        
        args = ["--schema"]
        if database:
            args.append(f"-D {database}")
        
        cmd = self._build_cmd(args)
        stdout, stderr, code = self._run_cmd(cmd)
        
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
        columns_tool = GetColumns(self.config, self.verbose, self.timeout)
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
