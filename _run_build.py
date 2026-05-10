"""Run build scripts with proper env vars (avoids cmd.exe trailing space issue)."""
import os
import sys
import subprocess

os.environ["ASHARE_MYSQL_USER"] = "cn_opr_red"
os.environ["ASHARE_MYSQL_PASSWORD"] = "sec_Bobo123"
os.environ["ASHARE_MYSQL_HOST"] = "localhost"
os.environ["ASHARE_MYSQL_PORT"] = "3306"
os.environ["ASHARE_MYSQL_DB"] = "cn_market_red"

if len(sys.argv) < 2:
    print("Usage: python _run_build.py <script.py> [args...]")
    sys.exit(1)

script = sys.argv[1]
script_args = sys.argv[2:]

# Only inject --db-user for scripts that accept it (those with build_engine() signature)
# Scripts using get_engine() from env vars (like build_stock_fundamental_daily.py) don't need it
SCRIPTS_WITH_DB_ARGS = {
    "build_stock_quality_score_daily.py",
    "build_unified_alpha_score_daily.py",
    "build_mainline_lifecycle_daily.py",
    "validate_mainline_lifecycle_daily.py",
    "validate_unified_alpha_score_daily.py",
}
script_name = os.path.basename(script)
db_user = os.environ.get("ASHARE_MYSQL_USER", "root")
if script_name in SCRIPTS_WITH_DB_ARGS and "--db-user" not in script_args:
    script_args = ["--db-user", db_user] + script_args

cmd = [sys.executable, script] + script_args
print(f"Running: {' '.join(cmd)}")
print(f"Env: ASHARE_MYSQL_USER={os.environ['ASHARE_MYSQL_USER']}")
print()

result = subprocess.run(cmd, capture_output=False)
sys.exit(result.returncode)
