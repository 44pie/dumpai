"""Data dumping tools for DumpAI."""
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import time

try:
    from .base import BaseTool, ToolResult
except ImportError:
    from base import BaseTool, ToolResult


def parse_sqlmap_table(output: str, expected_columns: List[str] = None) -> List[Dict]:
    """Parse SQLMap table output in multiple formats."""
    rows = []
    
    lines = output.split('\n')
    header_cols = []
    in_table = False
    header_found = False
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        is_separator = (line_stripped.startswith('+') and line_stripped.endswith('+') and 
                       '-' in line_stripped)
        
        if is_separator:
            if not header_found and i + 1 < len(lines):
                next_line = lines[i+1].strip()
                if next_line.startswith('|'):
                    header_cols = [c.strip() for c in next_line.split('|') if c.strip()]
                    header_found = True
                    in_table = True
            continue
        
        if in_table and line_stripped.startswith('|') and header_cols:
            values = [v.strip() for v in line_stripped.split('|') if v.strip()]
            if len(values) == len(header_cols) and values != header_cols:
                row = dict(zip(header_cols, values))
                rows.append(row)
    
    if rows:
        return rows
    
    retrieved = re.findall(r"\[INFO\] retrieved:\s*'([^']*)'", output)
    if retrieved and expected_columns:
        num_cols = len(expected_columns)
        if len(retrieved) >= num_cols:
            for i in range(0, len(retrieved), num_cols):
                chunk = retrieved[i:i+num_cols]
                if len(chunk) == num_cols:
                    row = dict(zip(expected_columns, chunk))
                    rows.append(row)
    
    return rows


class DumpTable(BaseTool):
    """Dump entire table."""
    
    name = "dump_table"
    description = "Dump all data from the specified table"
    
    def execute(self, database: str = None, table: str = None, 
                max_rows: int = 0, **kwargs) -> ToolResult:
        start = time.time()
        
        if not table:
            return ToolResult(success=False, error="Table name required")
        
        db = database or self.database
        
        args = []
        if db:
            args.append(f'-D {db}')
        args.append(f'-T {table}')
        args.append("--dump")
        if max_rows > 0:
            args.append(f"--stop {max_rows}")
        cmd = self._build_cmd(*args)
        
        output = self._run_cmd(cmd, idle_timeout=900)
        rows = parse_sqlmap_table(output)
        
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
    
    def _dump_single_column(self, database: str, table: str, column: str, 
                            max_rows: int = 0) -> Dict:
        """Dump a single column."""
        args = []
        if database:
            args.append(f'-D {database}')
        args.append(f'-T {table}')
        args.append(f'-C {column}')
        args.append("--dump")
        if max_rows > 0:
            args.append(f"--stop {max_rows}")
        cmd = self._build_cmd(*args)
        
        output = self._run_cmd(cmd, idle_timeout=600)
        rows = parse_sqlmap_table(output, [column])
        
        return {"column": column, "values": [r.get(column, "") for r in rows]}
    
    def execute(self, database: str = None, table: str = None, 
                columns: List[str] = None, max_rows: int = 0,
                parallel: bool = True, max_workers: int = 5, **kwargs) -> ToolResult:
        start = time.time()
        
        if not table:
            return ToolResult(success=False, error="Table name required")
        if not columns:
            return ToolResult(success=False, error="Columns list required")
        
        db = database or self.database
        
        if parallel and len(columns) > 1:
            if self.verbose:
                print(f"    [PARALLEL] Dumping {len(columns)} columns with {max_workers} workers")
            
            column_data = {}
            max_len = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._dump_single_column, db, table, col, max_rows): col 
                    for col in columns
                }
                for future in as_completed(futures):
                    col = futures[future]
                    try:
                        result = future.result()
                        column_data[col] = result["values"]
                        max_len = max(max_len, len(result["values"]))
                        if self.verbose:
                            print(f"    [PARALLEL] Column '{col}': {len(result['values'])} values")
                    except Exception as e:
                        column_data[col] = []
                        if self.verbose:
                            print(f"    [PARALLEL] Column '{col}' error: {e}")
            
            rows = []
            for i in range(max_len):
                row = {}
                for col in columns:
                    values = column_data.get(col, [])
                    row[col] = values[i] if i < len(values) else ""
                rows.append(row)
        else:
            args = []
            if db:
                args.append(f'-D {db}')
            args.append(f'-T {table}')
            args.append(f'-C {",".join(columns)}')
            args.append("--dump")
            if max_rows > 0:
                args.append(f"--stop {max_rows}")
            cmd = self._build_cmd(*args)
            
            output = self._run_cmd(cmd, idle_timeout=900)
            rows = parse_sqlmap_table(output, columns)
        
        return ToolResult(
            success=len(rows) > 0,
            data=rows,
            raw_output="",
            execution_time=time.time() - start,
            metadata={
                "database": db, 
                "table": table, 
                "columns": columns,
                "rows": len(rows),
                "parallel": parallel
            }
        )


class DumpColumnsParallel(BaseTool):
    """Dump columns from multiple tables in parallel."""
    
    name = "dump_columns_parallel"
    description = "Dump columns from multiple tables simultaneously"
    
    def _dump_table_columns(self, database: str, table: str, columns: List[str],
                            max_rows: int = 0) -> Dict:
        """Dump columns from a single table."""
        args = []
        if database:
            args.append(f'-D {database}')
        args.append(f'-T {table}')
        args.append(f'-C {",".join(columns)}')
        args.append("--dump")
        if max_rows > 0:
            args.append(f"--stop {max_rows}")
        cmd = self._build_cmd(*args)
        
        output = self._run_cmd(cmd, idle_timeout=900)
        rows = parse_sqlmap_table(output, columns)
        
        return {"table": table, "columns": columns, "rows": rows}
    
    def execute(self, database: str = None, 
                table_columns: Dict[str, List[str]] = None,
                max_rows: int = 0, max_workers: int = 5, **kwargs) -> ToolResult:
        """
        table_columns: {"users": ["email", "pass"], "customers": ["name", "phone"]}
        """
        start = time.time()
        
        if not table_columns:
            return ToolResult(success=False, error="table_columns dict required")
        
        db = database or self.database
        all_results = {}
        
        if self.verbose:
            print(f"    [PARALLEL] Dumping {len(table_columns)} tables with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._dump_table_columns, db, table, cols, max_rows): table 
                for table, cols in table_columns.items()
            }
            for future in as_completed(futures):
                table = futures[future]
                try:
                    result = future.result()
                    all_results[table] = result["rows"]
                    if self.verbose:
                        print(f"    [PARALLEL] Table '{table}': {len(result['rows'])} rows")
                except Exception as e:
                    all_results[table] = []
                    if self.verbose:
                        print(f"    [PARALLEL] Table '{table}' error: {e}")
        
        total_rows = sum(len(rows) for rows in all_results.values())
        
        return ToolResult(
            success=total_rows > 0,
            data=all_results,
            raw_output="",
            execution_time=time.time() - start,
            metadata={
                "database": db,
                "tables": list(table_columns.keys()),
                "total_rows": total_rows
            }
        )
