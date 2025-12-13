"""Data dumping tools for DumpAI."""
from typing import Dict, List, Optional
import re
import time

try:
    from .base import BaseTool, ToolResult
except ImportError:
    from base import BaseTool, ToolResult


class DumpTable(BaseTool):
    """Dump entire table."""
    
    name = "dump_table"
    description = "Dump all data from the specified table"
    
    def _parse_dump(self, output: str) -> List[Dict]:
        """Parse dumped data from SQLMap output."""
        rows = []
        
        table_match = re.search(r'Table:\s*(\S+)', output)
        if not table_match:
            return rows
        
        header_match = re.search(r'\+-+\+[\s\S]*?\|\s*(.+?)\s*\|\s*\n\+-+\+', output)
        if not header_match:
            return rows
        
        header_line = header_match.group(1)
        columns = [c.strip() for c in re.split(r'\s*\|\s*', header_line) if c.strip()]
        
        data_section = output[header_match.end():]
        for line in data_section.split('\n'):
            if line.startswith('|') and not line.startswith('+-'):
                values = [v.strip() for v in re.split(r'\s*\|\s*', line) if v.strip()]
                if len(values) == len(columns):
                    row = dict(zip(columns, values))
                    rows.append(row)
        
        return rows
    
    def execute(self, database: str = None, table: str = None, **kwargs) -> ToolResult:
        start = time.time()
        
        if not table:
            return ToolResult(success=False, error="Table name required")
        
        db = database or self.database
        
        parts = self._build_base_cmd()
        if db:
            parts.append(f'-D {db}')
        parts.append(f'-T {table}')
        parts.append("--dump")
        cmd = " ".join(parts)
        
        output = self._run_cmd(cmd, idle_timeout=900)
        rows = self._parse_dump(output)
        
        return ToolResult(
            success=len(rows) > 0,
            data=rows,
            raw_output=output,
            execution_time=time.time() - start,
            metadata={"database": db, "table": table, "rows": len(rows)}
        )


class DumpColumns(BaseTool):
    """Dump specific columns from a table."""
    
    name = "dump_columns"
    description = "Dump specific columns from the specified table"
    
    def _parse_dump(self, output: str) -> List[Dict]:
        """Parse dumped data from SQLMap output."""
        rows = []
        
        header_match = re.search(r'\+-+\+[\s\S]*?\|\s*(.+?)\s*\|\s*\n\+-+\+', output)
        if not header_match:
            return rows
        
        header_line = header_match.group(1)
        columns = [c.strip() for c in re.split(r'\s*\|\s*', header_line) if c.strip()]
        
        data_section = output[header_match.end():]
        for line in data_section.split('\n'):
            if line.startswith('|') and not line.startswith('+-'):
                values = [v.strip() for v in re.split(r'\s*\|\s*', line) if v.strip()]
                if len(values) == len(columns):
                    row = dict(zip(columns, values))
                    rows.append(row)
        
        return rows
    
    def execute(self, database: str = None, table: str = None, 
                columns: List[str] = None, **kwargs) -> ToolResult:
        start = time.time()
        
        if not table:
            return ToolResult(success=False, error="Table name required")
        if not columns:
            return ToolResult(success=False, error="Columns list required")
        
        db = database or self.database
        
        parts = self._build_base_cmd()
        if db:
            parts.append(f'-D {db}')
        parts.append(f'-T {table}')
        parts.append(f'-C {",".join(columns)}')
        parts.append("--dump")
        cmd = " ".join(parts)
        
        output = self._run_cmd(cmd, idle_timeout=900)
        rows = self._parse_dump(output)
        
        return ToolResult(
            success=len(rows) > 0,
            data=rows,
            raw_output=output,
            execution_time=time.time() - start,
            metadata={
                "database": db, 
                "table": table, 
                "columns": columns,
                "rows": len(rows)
            }
        )
