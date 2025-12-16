# DumpAI v3.0 - Technical Architecture Report

## Overview

DumpAI is an AI-powered autonomous SQL injection data extractor following hackingBuddyGPT architecture patterns. It uses a **Reason → Act → Observe → Adapt** cycle for intelligent exploitation.

---

## Core Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DumpAI Agent                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌─────────────┐ │
│  │ Console │    │ Planner │    │ Strategy │    │   Memory    │ │
│  │  (UI)   │◄──►│  (AI)   │◄──►│ Manager  │◄──►│ (State)     │ │
│  └─────────┘    └─────────┘    └──────────┘    └─────────────┘ │
│       │              │              │                │          │
│       │              ▼              │                │          │
│       │        ┌──────────┐        │                │          │
│       │        │  Tools   │◄───────┴────────────────┘          │
│       │        │ (SQLMap) │                                     │
│       │        └──────────┘                                     │
└───────┴─────────────────────────────────────────────────────────┘
```

---

## Execution Stages

### PHASE 1: Injection Analysis
**File:** `agent_v3.py` lines 219-253

1. Execute `enumerate_dbs` tool with SQLMap
2. Parse output to detect injection characteristics
3. AI analyzes injection type, speed, WAF presence
4. Determine strategy: `full_enumeration` or `smart_search`
5. Store detected techniques (U/E/S/B/T) for command optimization

### PHASE 2: Database Selection
**File:** `agent_v3.py` lines 255-277

1. Filter system databases (information_schema, mysql, etc.)
2. AI or automatic selection of target database
3. Store in memory for subsequent phases

### PHASE 3: Table Discovery
**File:** `agent_v3.py` lines 279-310

1. **Smart Search** (if slow injection): Search by patterns
2. **Full Enumeration** (if fast injection): Get all tables
3. Parse SQLMap output for table list

### PHASE 4: AI CMS Detection & Prioritization
**File:** `agent_v3.py` lines 312-340

1. AI detects CMS (PrestaShop, WordPress, Magento)
2. AI prioritizes tables by utility score (0.0-1.0)
3. Filter tables by requested categories
4. Generate extraction plan

### PHASE 5: Parallel Extraction
**File:** `agent_v3.py` lines 342-410

1. Get columns for each prioritized table
2. AI selects valuable columns per category
3. **Parallel dump**: Multiple tables simultaneously
4. Merge results and save to output directory

---

## AI Decision Points

### 1. Injection Analysis AI
**File:** `planner.py` lines 160-218

**Prompt:**
```
Analyze SQLMap output to determine injection type and optimal strategy.

Output JSON with:
- injection_type: time_based|boolean_blind|error_based|union|stacked
- is_slow: true if ONLY blind/time-based available
- recommended_strategy: full_enumeration or smart_search
- waf_detected: true if WAF indicators present
- tampers_suggested: list of bypass tamper scripts
```

**Decision Logic:**
- UNION/Error-based/Stacked available → `full_enumeration` (fast)
- Only Boolean/Time-based → `smart_search` (slow, targeted queries)

### 2. Table Prioritization AI
**File:** `planner.py` lines 289-380

**Prompt:**
```
Prioritize database tables for extraction.

CMS: {detected_cms}
CATEGORIES REQUESTED: {user_categories}
TABLES: {discovered_tables}

CATEGORY DEFINITIONS:
- user_data: employee/admin/staff accounts (NOT customers)
- customer_data: customer/client personal info, addresses, orders
- api_key: API keys, tokens, webservice credentials
- sys_data: configuration, settings
- email_pass: email + password combinations

Return JSON list with score (0.0-1.0), category, reason for each table.
```

**Decision Logic:**
- Score 0.9+: High-value tables (admin, employee, credentials)
- Score 0.5-0.9: Medium value (configuration, tokens)
- Score <0.5: Low priority (logs, cache)
- Tables filtered by requested categories

### 3. Column Selection AI
**File:** `planner.py` lines 382-450

**Prompt:**
```
Prioritize columns for extraction from table {table_name}.

COLUMNS: {column_list}
CATEGORIES: {user_categories}

Return JSON with columns grouped by category:
- user_data: email, password, username columns
- api_key: token, key, secret columns
- customer_data: name, address, phone columns
```

### 4. Error Recovery AI
**File:** `planner.py` lines 490-530

**Prompt:**
```
SQLMap command failed with output:
{error_output}

Previous attempts: {retry_history}

Suggest fix:
- technique_override: different technique to try
- tamper_add: WAF bypass tamper to add
- skip: skip this table
- abort: critical failure
```

---

## Strategy Management

**File:** `strategy.py`

### Strategies:
1. **full_enumeration**: Use with UNION/Error-based injections
   - Fast data extraction
   - Full table dumps with `--dump`
   
2. **smart_search**: Use with blind/time-based only
   - Search tables by pattern (`--search -T pattern`)
   - Dump specific columns only
   - Minimize queries for slow injections

### Strategy Selection Logic:
```python
if "union" in available or "error_based" in available or "stacked" in available:
    strategy = "full_enumeration"
else:
    strategy = "smart_search"
```

---

## Technique Optimization

**File:** `tools.py` lines 219-238

DumpAI automatically optimizes `--technique` flag:

1. **Detection**: Parse available techniques from SQLMap session
2. **Filtering**: Keep only techniques that actually work
3. **Ordering**: Prioritize by speed (U > E > S > Q > B > T)

**Example:**
```
Input:  --technique=TBEUSQ
Output: --technique=ST  (if only Stacked and Time-based available)
```

**Priority Order:**
| Code | Technique | Speed |
|------|-----------|-------|
| U | UNION query | Fastest |
| E | Error-based | Fast |
| S | Stacked queries | Medium |
| Q | Inline queries | Medium |
| B | Boolean blind | Slow |
| T | Time-based blind | Slowest |

---

## Memory Management

**File:** `memory.py`

### Stored State:
- `databases`: Discovered databases
- `current_database`: Selected target
- `tables`: Discovered tables
- `columns_cache`: Column info per table
- `cms_detected`: Detected CMS type
- `injection_type`: Primary injection type
- `available_techniques`: Working technique codes (e.g., "ST")
- `hypotheses`: AI beliefs about target
- `extracted_data`: Extracted rows by category
- `stats`: AI calls, tokens, timing

### Thread Safety:
All memory operations use `threading.Lock()` for parallel extraction safety.

---

## Output Structure

**Output Directory:** (specified by `-o` flag)

```
output_dir/
├── users.json          # Per-table extracted data
├── employees.json
├── ...
└── dump_all.json       # All tables combined
```

**JSON Format:**
```json
{
  "users": [
    {"email": "admin@site.com", "pass": "hash123", "name": "Admin"},
    ...
  ],
  "employees": [...]
}
```

---

## Command Line Interface

```bash
dumpai.py -c "sqlmap command" [options]

Required:
  -c, --cmd        Base SQLMap command with request file

Options:
  -o, --output     Output directory for extracted data
  -cat, --categories  Categories to extract (user_data,api_key,sys_data,customer_data,email_pass)
  -v, --verbose    Verbosity level (0=minimal, 1=AI reasoning, 2=debug)
  -mr, --max-rows  Maximum rows to dump per table
  -p, --parallel   Max parallel workers for extraction
  --no-parallel    Disable parallel extraction
```

---

## Version History

### v3.0.10 (2025-12-16)
- Fixed: available_techniques now always populated from LLM or fallback
- Added log output showing detected techniques at verbosity >= 1

### v3.0.9 (2025-12-16)
- Filter `--technique` to only available (detected) techniques
- Preserve user's `--output-dir` for SQLMap cached session
- Auto-optimize technique order (fast first)

### v3.0.8 (2025-12-16)
- Fixed session caching: SQLMap reuses detected injection
- Removed temp directory override for enumeration phases

### v3.0.7 (2025-12-16)
- Fixed extracted data display in summary
- Fixed memory data structure for categories

### v3.0.0 (2025-12-15)
- Complete rewrite following hackingBuddyGPT patterns
- Full AI integration at every decision point
- Parallel extraction with thread-safe memory
- Rich console output with AI reasoning visibility
