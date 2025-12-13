"""Base Tool class for DumpAI."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import subprocess
import select
import time
import re


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    data: Any = None
    error: str = ""
    raw_output: str = ""
    execution_time: float = 0.0
    metadata: Dict = field(default_factory=dict)


class BaseTool(ABC):
    """Base class for all DumpAI tools."""
    
    name: str = "base_tool"
    description: str = "Base tool"
    
    def __init__(self, sqlmap_config: Dict, verbose: bool = False):
        self.config = sqlmap_config
        self.prefix = sqlmap_config.get("prefix", "")
        self.sqlmap_path = sqlmap_config.get("sqlmap_path", "sqlmap")
        self.request_file = sqlmap_config.get("request_file", "")
        self.parameter = sqlmap_config.get("parameter", "")
        self.database = sqlmap_config.get("database", "")
        self.extra_flags = sqlmap_config.get("extra_flags", [])
        self.verbose = verbose
    
    def _build_base_cmd(self) -> List[str]:
        """Build base SQLMap command parts."""
        parts = []
        
        if self.prefix:
            parts.append(self.prefix.strip())
        
        parts.append(self.sqlmap_path)
        parts.append(f'-r "{self.request_file}"')
        
        if self.parameter:
            parts.append(f'-p "{self.parameter}"')
        
        parts.extend(self.extra_flags)
        
        return parts
    
    SQLMAP_KEY_PATTERNS = [
        "available databases",
        "fetching tables",
        "fetching columns",
        "retrieved:",
        "[INFO] the back-end DBMS is",
        "time-based blind",
        "boolean-based blind",
        "error-based",
        "UNION query",
        "stacked queries",
        "[WARNING]",
        "[ERROR]",
        "[CRITICAL]",
        "might be injectable",
        "is vulnerable",
        "sqlmap identified",
        "fetched data logged",
        "dumping",
        "entries found"
    ]
    
    def _run_cmd(self, cmd: str, timeout: int = 0, idle_timeout: int = 600) -> str:
        """Execute command with streaming output."""
        if self.verbose:
            print(f"[*] CMD: {cmd}")
        
        start_time = time.time()
        
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            output_lines = []
            last_output_time = time.time()
            
            while True:
                if process.stdout:
                    ready, _, _ = select.select([process.stdout], [], [], 1.0)
                    
                    if ready:
                        line = process.stdout.readline()
                        if line:
                            output_lines.append(line)
                            last_output_time = time.time()
                            
                            if self.verbose:
                                self._check_key_output(line)
                
                if process.poll() is not None:
                    remaining = process.stdout.read() if process.stdout else ""
                    if remaining:
                        output_lines.append(remaining)
                    break
                
                if time.time() - last_output_time > idle_timeout:
                    process.kill()
                    break
                
                if timeout > 0 and time.time() - start_time > timeout:
                    process.kill()
                    break
            
            return "".join(output_lines)
            
        except Exception as e:
            return f"ERROR: {e}"
    
    def _check_key_output(self, line: str):
        """Print key SQLMap output lines in verbose mode."""
        line_lower = line.lower()
        for pattern in self.SQLMAP_KEY_PATTERNS:
            if pattern.lower() in line_lower:
                clean = line.strip()
                if clean:
                    print(f"    [SQLMAP] {clean}")
                break
    
    def _parse_table_output(self, output: str) -> List[str]:
        """Parse table/column names from SQLMap output."""
        items = []
        
        matches = re.findall(r'\|\s*([a-zA-Z0-9_]+)\s*\|', output)
        items = [t for t in matches if t.lower() not in ['table', 'tables', 'column', 'columns', 'type', 'name']]
        
        if not items:
            matches = re.findall(r'\[\*\]\s+(\S+)', output)
            items = [m for m in matches if not any(x in m.lower() for x in ['fetching', 'using'])]
        
        return list(dict.fromkeys(items))
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool. Must be implemented by subclasses."""
        pass
    
    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.description}>"
