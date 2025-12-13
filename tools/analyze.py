"""AI Analysis tools for DumpAI."""
from typing import Dict, List, Optional
import json
import os
import requests
import time

try:
    from .base import BaseTool, ToolResult
except ImportError:
    from base import BaseTool, ToolResult

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"


class AnalyzeSchema(BaseTool):
    """AI-powered schema analysis."""
    
    name = "analyze_schema"
    description = "Use AI to analyze database schema and identify valuable tables"
    
    def _call_ai(self, prompt: str, system_prompt: str) -> Dict:
        """Call OpenAI API."""
        if not OPENAI_API_KEY:
            return {}
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 4000,
                    "response_format": {"type": "json_object"}
                },
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return json.loads(content)
            return {}
        except Exception:
            return {}
    
    def execute(self, database: str = None, tables: List[str] = None,
                categories: List[str] = None, **kwargs) -> ToolResult:
        start = time.time()
        
        if not tables:
            return ToolResult(success=False, error="Tables list required")
        
        system_prompt = """You are a database security analyst AI for penetration testing.
Analyze database schema and identify valuable data for extraction.

CATEGORIES:
1. user_data: Admin/employee accounts, passwords, hashes, tokens
2. customer_data: Customer PII (names, addresses, phones, emails)
3. email_pass: Email + password/hash pairs
4. api_key: API keys, secrets, tokens
5. sys_data: System access (DB creds, phpMyAdmin, FTP, SSH)

Always respond with valid JSON only."""

        categories_text = ", ".join(categories) if categories else "all"
        
        prompt = f"""Analyze this database schema:

DATABASE: {database or 'unknown'}
TABLES: {', '.join(tables)}
CATEGORIES TO FIND: {categories_text}

Detect the CMS (PrestaShop, WordPress, Magento, etc.) from table prefixes.
For each category, identify relevant tables with columns to extract.

Response format:
{{
    "cms_detected": "CMS name or Unknown",
    "database_type": "e-commerce|blog|corporate|custom",
    "extractions": [
        {{
            "category": "category_name",
            "table": "table_name",
            "columns": ["col1", "col2"],
            "priority": "high|medium|low",
            "reason": "why this table is valuable"
        }}
    ]
}}"""

        result = self._call_ai(prompt, system_prompt)
        
        return ToolResult(
            success=bool(result),
            data=result,
            execution_time=time.time() - start,
            metadata={"tables_analyzed": len(tables)}
        )


class AnalyzeColumns(BaseTool):
    """AI-powered column analysis."""
    
    name = "analyze_columns"
    description = "Use AI to categorize table columns for extraction"
    
    def _call_ai(self, prompt: str, system_prompt: str) -> Dict:
        """Call OpenAI API."""
        if not OPENAI_API_KEY:
            return {}
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "response_format": {"type": "json_object"}
                },
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return json.loads(content)
            return {}
        except Exception:
            return {}
    
    def execute(self, table: str = None, columns: List[str] = None,
                categories: List[str] = None, **kwargs) -> ToolResult:
        start = time.time()
        
        if not table or not columns:
            return ToolResult(success=False, error="Table and columns required")
        
        system_prompt = """You are a database security analyst.
Categorize columns for data extraction. Consider multiple languages.

password/pwd/pass/hash/passwd = credentials
email/mail/correo = email
api_key/secret/token = API keys
host/server/connection = system data
phone/address/name = customer PII

Respond with JSON only."""

        categories_text = ", ".join(categories) if categories else "all"
        
        prompt = f"""Categorize these columns:

TABLE: {table}
COLUMNS: {', '.join(columns)}
CATEGORIES: {categories_text}

Response format:
{{
    "recommended_extractions": [
        {{
            "columns": ["col1", "col2"],
            "category": "category_name",
            "priority": "high|medium|low"
        }}
    ]
}}"""

        result = self._call_ai(prompt, system_prompt)
        
        return ToolResult(
            success=bool(result),
            data=result,
            execution_time=time.time() - start,
            metadata={"table": table, "columns_analyzed": len(columns)}
        )
