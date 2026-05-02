-- ═══════════════════════════════════════════════════════════
-- TPC-H Schema Initialization (SF=1)
-- PostgreSQL 17 compatible
-- ═══════════════════════════════════════════════════════════

-- Enable pg_hint_plan extension
CREATE EXTENSION IF NOT EXISTS pg_hint_plan;

-- ── NATION ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS nation (
    n_nationkey  INTEGER NOT NULL PRIMARY KEY,
    n_name       CHAR(25) NOT NULL,
    n_regionkey  INTEGER NOT NULL,
    n_comment    VARCHAR(152)
);

-- ── REGION ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS region (
    r_regionkey  INTEGER NOT NULL PRIMARY KEY,
    r_name       CHAR(25) NOT NULL,
    r_comment    VARCHAR(152)
);

-- ── PART ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS part (
    p_partkey     INTEGER NOT NULL PRIMARY KEY,
    p_name        VARCHAR(55) NOT NULL,
    p_mfgr        CHAR(25) NOT NULL,
    p_brand       CHAR(10) NOT NULL,
    p_type        VARCHAR(25) NOT NULL,
    p_size        INTEGER NOT NULL,
    p_container   CHAR(10) NOT NULL,
    p_retailprice DECIMAL(15,2) NOT NULL,
    p_comment     VARCHAR(23) NOT NULL
);

-- ── SUPPLIER ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS supplier (
    s_suppkey   INTEGER NOT NULL PRIMARY KEY,
    s_name      CHAR(25) NOT NULL,
    s_address   VARCHAR(40) NOT NULL,
    s_nationkey INTEGER NOT NULL,
    s_phone     CHAR(15) NOT NULL,
    s_acctbal   DECIMAL(15,2) NOT NULL,
    s_comment   VARCHAR(101) NOT NULL
);

-- ── PARTSUPP ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS partsupp (
    ps_partkey    INTEGER NOT NULL,
    ps_suppkey    INTEGER NOT NULL,
    ps_availqty   INTEGER NOT NULL,
    ps_supplycost DECIMAL(15,2) NOT NULL,
    ps_comment    VARCHAR(199) NOT NULL,
    PRIMARY KEY (ps_partkey, ps_suppkey)
);

-- ── CUSTOMER ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS customer (
    c_custkey    INTEGER NOT NULL PRIMARY KEY,
    c_name       VARCHAR(25) NOT NULL,
    c_address    VARCHAR(40) NOT NULL,
    c_nationkey  INTEGER NOT NULL,
    c_phone      CHAR(15) NOT NULL,
    c_acctbal    DECIMAL(15,2) NOT NULL,
    c_mktsegment CHAR(10) NOT NULL,
    c_comment    VARCHAR(117) NOT NULL
);

-- ── ORDERS ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS orders (
    o_orderkey      INTEGER NOT NULL PRIMARY KEY,
    o_custkey       INTEGER NOT NULL,
    o_orderstatus   CHAR(1) NOT NULL,
    o_totalprice    DECIMAL(15,2) NOT NULL,
    o_orderdate     DATE NOT NULL,
    o_orderpriority CHAR(15) NOT NULL,
    o_clerk         CHAR(15) NOT NULL,
    o_shippriority  INTEGER NOT NULL,
    o_comment       VARCHAR(79) NOT NULL
);

-- ── LINEITEM ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS lineitem (
    l_orderkey      INTEGER NOT NULL,
    l_partkey       INTEGER NOT NULL,
    l_suppkey       INTEGER NOT NULL,
    l_linenumber    INTEGER NOT NULL,
    l_quantity      DECIMAL(15,2) NOT NULL,
    l_extendedprice DECIMAL(15,2) NOT NULL,
    l_discount      DECIMAL(15,2) NOT NULL,
    l_tax           DECIMAL(15,2) NOT NULL,
    l_returnflag    CHAR(1) NOT NULL,
    l_linestatus    CHAR(1) NOT NULL,
    l_shipdate      DATE NOT NULL,
    l_commitdate    DATE NOT NULL,
    l_receiptdate   DATE NOT NULL,
    l_shipinstruct  CHAR(25) NOT NULL,
    l_shipmode      CHAR(10) NOT NULL,
    l_comment       VARCHAR(44) NOT NULL,
    PRIMARY KEY (l_orderkey, l_linenumber)
);

-- ═══════════════════════════════════════════════════════════
-- FOREIGN KEYS
-- ═══════════════════════════════════════════════════════════

ALTER TABLE nation ADD CONSTRAINT fk_nation_region
    FOREIGN KEY (n_regionkey) REFERENCES region(r_regionkey);

ALTER TABLE supplier ADD CONSTRAINT fk_supplier_nation
    FOREIGN KEY (s_nationkey) REFERENCES nation(n_nationkey);

ALTER TABLE customer ADD CONSTRAINT fk_customer_nation
    FOREIGN KEY (c_nationkey) REFERENCES nation(n_nationkey);

ALTER TABLE partsupp ADD CONSTRAINT fk_partsupp_part
    FOREIGN KEY (ps_partkey) REFERENCES part(p_partkey);

ALTER TABLE partsupp ADD CONSTRAINT fk_partsupp_supplier
    FOREIGN KEY (ps_suppkey) REFERENCES supplier(s_suppkey);

ALTER TABLE orders ADD CONSTRAINT fk_orders_customer
    FOREIGN KEY (o_custkey) REFERENCES customer(c_custkey);

ALTER TABLE lineitem ADD CONSTRAINT fk_lineitem_orders
    FOREIGN KEY (l_orderkey) REFERENCES orders(o_orderkey);

ALTER TABLE lineitem ADD CONSTRAINT fk_lineitem_partsupp
    FOREIGN KEY (l_partkey, l_suppkey) REFERENCES partsupp(ps_partkey, ps_suppkey);

-- ═══════════════════════════════════════════════════════════
-- INDEXES (critical for query optimization benchmarking)
-- ═══════════════════════════════════════════════════════════

-- Lineitem indexes
CREATE INDEX IF NOT EXISTS idx_lineitem_orderkey ON lineitem(l_orderkey);
CREATE INDEX IF NOT EXISTS idx_lineitem_partkey ON lineitem(l_partkey);
CREATE INDEX IF NOT EXISTS idx_lineitem_suppkey ON lineitem(l_suppkey);
CREATE INDEX IF NOT EXISTS idx_lineitem_shipdate ON lineitem(l_shipdate);
CREATE INDEX IF NOT EXISTS idx_lineitem_commitdate ON lineitem(l_commitdate);
CREATE INDEX IF NOT EXISTS idx_lineitem_receiptdate ON lineitem(l_receiptdate);
CREATE INDEX IF NOT EXISTS idx_lineitem_partsupp ON lineitem(l_partkey, l_suppkey);

-- Orders indexes
CREATE INDEX IF NOT EXISTS idx_orders_custkey ON orders(o_custkey);
CREATE INDEX IF NOT EXISTS idx_orders_orderdate ON orders(o_orderdate);

-- Customer indexes
CREATE INDEX IF NOT EXISTS idx_customer_nationkey ON customer(c_nationkey);
CREATE INDEX IF NOT EXISTS idx_customer_mktsegment ON customer(c_mktsegment);

-- Supplier indexes
CREATE INDEX IF NOT EXISTS idx_supplier_nationkey ON supplier(s_nationkey);

-- Part indexes
CREATE INDEX IF NOT EXISTS idx_part_type ON part(p_type);
CREATE INDEX IF NOT EXISTS idx_part_brand ON part(p_brand);
CREATE INDEX IF NOT EXISTS idx_part_size ON part(p_size);

-- Partsupp indexes
CREATE INDEX IF NOT EXISTS idx_partsupp_suppkey ON partsupp(ps_suppkey);
CREATE INDEX IF NOT EXISTS idx_partsupp_partkey ON partsupp(ps_partkey);

-- Nation/Region indexes (small tables but referenced frequently)
CREATE INDEX IF NOT EXISTS idx_nation_regionkey ON nation(n_regionkey);

-- ═══════════════════════════════════════════════════════════
-- ANALYZE (populate pg_statistic for the RL agent)
-- ═══════════════════════════════════════════════════════════
-- Note: ANALYZE should be run after data is loaded.
-- If using dbgen to load data, run ANALYZE after COPY commands.

-- Placeholder: uncomment after data load
-- ANALYZE nation;
-- ANALYZE region;
-- ANALYZE part;
-- ANALYZE supplier;
-- ANALYZE partsupp;
-- ANALYZE customer;
-- ANALYZE orders;
-- ANALYZE lineitem;

-- ═══════════════════════════════════════════════════════════
-- VERIFY pg_hint_plan is loaded
-- ═══════════════════════════════════════════════════════════
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_hint_plan') THEN
        RAISE NOTICE 'pg_hint_plan extension loaded successfully';
    ELSE
        RAISE WARNING 'pg_hint_plan extension NOT loaded — check shared_preload_libraries';
    END IF;
END $$;
