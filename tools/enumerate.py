"""Enumeration tools for DumpAI."""
from typing import List, Optional, Dict
import time
import re

try:
    from .base import BaseTool, ToolResult
except ImportError:
    from base import BaseTool, ToolResult


class EnumerateDBs(BaseTool):
    """Enumerate available databases."""
    
    name = "enumerate_dbs"
    description = "List all available databases on the target"
    
    def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        
        cmd = self._build_cmd("--dbs")
        
        output = self._run_cmd(cmd, idle_timeout=300)
        databases = self._parse_table_output(output)
        
        return ToolResult(
            success=len(databases) > 0,
            data=databases,
            raw_output=output,
            execution_time=time.time() - start,
            metadata={"count": len(databases)}
        )


class EnumerateTables(BaseTool):
    """Enumerate tables in a database."""
    
    name = "enumerate_tables"
    description = "List all tables in the specified database"
    
    def execute(self, database: Optional[str] = None, **kwargs) -> ToolResult:
        start = time.time()
        
        db = database or self.database
        
        args = []
        if db:
            args.append(f'-D {db}')
        args.append("--tables")
        cmd = self._build_cmd(*args)
        
        output = self._run_cmd(cmd, idle_timeout=600)
        tables = self._parse_table_output(output)
        
        return ToolResult(
            success=len(tables) > 0,
            data=tables,
            raw_output=output,
            execution_time=time.time() - start,
            metadata={"database": db, "count": len(tables)}
        )


class GetColumns(BaseTool):
    """Get columns for a table."""
    
    name = "get_columns"
    description = "Get all columns for the specified table"
    
    def execute(self, database: Optional[str] = None, table: Optional[str] = None, **kwargs) -> ToolResult:
        start = time.time()
        
        if not table:
            return ToolResult(success=False, error="Table name required")
        
        db = database or self.database
        
        args = []
        if db:
            args.append(f'-D {db}')
        args.append(f'-T {table}')
        args.append("--columns")
        cmd = self._build_cmd(*args)
        
        output = self._run_cmd(cmd, idle_timeout=600)
        columns = self._parse_table_output(output)
        
        return ToolResult(
            success=len(columns) > 0,
            data=columns,
            raw_output=output,
            execution_time=time.time() - start,
            metadata={"database": db, "table": table, "count": len(columns)}
        )


class SearchTables(BaseTool):
    """Search for tables by pattern - optimized for blind/time-based injections."""
    
    name = "search_tables"
    description = "Search for tables matching patterns (faster than full enumeration)"
    
    def execute(self, patterns: Optional[List[str]] = None, **kwargs) -> ToolResult:
        start = time.time()
        
        if not patterns:
            return ToolResult(success=False, error="Patterns list required")
        
        pattern_str = ",".join(patterns)
        
        cmd = self._build_cmd("--search", f'-T "{pattern_str}"')
        
        output = self._run_cmd(cmd, idle_timeout=900)
        tables = self._parse_search_output(output)
        
        return ToolResult(
            success=len(tables) > 0,
            data=tables,
            raw_output=output,
            execution_time=time.time() - start,
            metadata={"patterns": patterns, "count": len(tables)}
        )
    
    def _parse_search_output(self, output: str) -> List[str]:
        """Parse --search output to extract table names."""
        tables: List[str] = []
        
        matches = re.findall(r"found in database[s]?[^']*'([^']+)'.*?table[s]?:\s*'([^']+)'", output, re.IGNORECASE | re.DOTALL)
        for db, table in matches:
            tables.append(f"{db}.{table}")
        
        simple_matches = re.findall(r"\[\*\]\s+(\S+\.\S+)", output)
        for match in simple_matches:
            if match not in tables and '.' in match:
                tables.append(match)
        
        table_matches = re.findall(r"Table:\s*(\S+)", output)
        for t in table_matches:
            if t not in tables:
                tables.append(t)
        
        return list(dict.fromkeys(tables))


class SearchColumns(BaseTool):
    """Search for columns by pattern - optimized for blind/time-based injections."""
    
    name = "search_columns"
    description = "Search for columns matching patterns (faster than full enumeration)"
    
    def execute(self, patterns: Optional[List[str]] = None, **kwargs) -> ToolResult:
        start = time.time()
        
        if not patterns:
            return ToolResult(success=False, error="Patterns list required")
        
        pattern_str = ",".join(patterns)
        
        cmd = self._build_cmd("--search", f'-C "{pattern_str}"')
        
        output = self._run_cmd(cmd, idle_timeout=900)
        columns = self._parse_search_output(output)
        
        return ToolResult(
            success=len(columns) > 0,
            data=columns,
            raw_output=output,
            execution_time=time.time() - start,
            metadata={"patterns": patterns, "count": len(columns)}
        )
    
    def _parse_search_output(self, output: str) -> List[Dict]:
        """Parse --search output to extract column locations."""
        results: List[Dict] = []
        
        matches = re.findall(
            r"column[s]?\s+'([^']+)'\s+.*?found in.*?database[s]?\s+'([^']+)'.*?table[s]?\s+'([^']+)'",
            output, re.IGNORECASE | re.DOTALL
        )
        for col, db, table in matches:
            results.append({"column": col, "database": db, "table": table})
        
        simple_matches = re.findall(r"\[\*\]\s+(\S+)\.(\S+)\.(\S+)", output)
        for db, table, col in simple_matches:
            entry = {"column": col, "database": db, "table": table}
            if entry not in results:
                results.append(entry)
        
        return results
