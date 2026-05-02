#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
# load_tpch.sh — Download, build, and load TPC-H data (SF=1)
# ═══════════════════════════════════════════════════════════
#
# Usage:
#   chmod +x scripts/load_tpch.sh
#   ./scripts/load_tpch.sh
#
# Prerequisites:
#   - Docker container "querymind-postgres" running
#   - gcc/make available (for building dbgen)
#
# ═══════════════════════════════════════════════════════════

set -euo pipefail

SCALE_FACTOR="${SCALE_FACTOR:-1}"
TPCH_DIR="/tmp/tpch-kit"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5434}"
DB_USER="${DB_USER:-querymind}"
DB_NAME="${DB_NAME:-tpch}"
PGPASSWORD="${PGPASSWORD:-querymind}"
export PGPASSWORD

echo "═══════════════════════════════════════"
echo " TPC-H Data Loader (SF=${SCALE_FACTOR})"
echo "═══════════════════════════════════════"

# ── Step 1: Clone tpch-kit ──────────────────────────────────
if [ ! -d "$TPCH_DIR" ]; then
    echo "[1/5] Cloning tpch-kit..."
    git clone https://github.com/gregrahn/tpch-kit.git "$TPCH_DIR"
else
    echo "[1/5] tpch-kit already exists at $TPCH_DIR"
fi

# ── Step 2: Build dbgen ─────────────────────────────────────
echo "[2/5] Building dbgen..."
cd "$TPCH_DIR/dbgen"
make -f makefile.suite DATABASE=POSTGRESQL MACHINE=LINUX WORKLOAD=TPCH 2>/dev/null || make

# ── Step 3: Generate data ───────────────────────────────────
echo "[3/5] Generating TPC-H data (SF=${SCALE_FACTOR})..."
./dbgen -s "$SCALE_FACTOR" -f

echo "    Generated files:"
ls -lh *.tbl | awk '{print "      " $NF ": " $5}'

# ── Step 4: Load data into PostgreSQL ───────────────────────
echo "[4/5] Loading data into PostgreSQL..."

TABLES="region nation part supplier partsupp customer orders lineitem"

for table in $TABLES; do
    echo "    Loading $table..."
    PGOPTIONS="-c statement_timeout=0" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -q -c \
        "\\COPY $table FROM '$TPCH_DIR/dbgen/$table.tbl' WITH (FORMAT csv, DELIMITER '|')"
done

# ── Step 5: Analyze ─────────────────────────────────────────
echo "[5/5] Running ANALYZE to populate statistics..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "ANALYZE;"

echo ""
echo "═══════════════════════════════════════"
echo " TPC-H data loaded successfully!"
echo "═══════════════════════════════════════"

# Print table row counts
echo ""
echo "Table row counts:"
for table in $TABLES; do
    count=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c \
        "SELECT COUNT(*) FROM $table;")
    printf "  %-12s %s\n" "$table" "$count"
done
