"""Integration test: pipeline dry run on synthetic data."""
import pytest, sys, json
sys.path.insert(0, '.')
from benchmark.dtah_bench import DTAHBench

def test_dtah_bench_loads_tier1():
    bench = DTAHBench()
    prompts = bench.load_tier(1)
    assert len(prompts) == 50
    assert all("prompt" in p for p in prompts)
    assert all("ifc_entity" in p for p in prompts)

def test_dtah_bench_loads_tier2():
    bench = DTAHBench()
    prompts = bench.load_tier(2)
    assert len(prompts) == 50
    assert all("relations" in p for p in prompts)

def test_dtah_bench_loads_tier3():
    bench = DTAHBench()
    prompts = bench.load_tier(3)
    assert len(prompts) >= 20  # Will be 50 when complete; currently 30
    assert all("topology_constraints" in p for p in prompts)

def test_pilot_mode():
    bench = DTAHBench(pilot_mode=True)
    pilot = bench.load_pilot()
    assert len(pilot) == 50  # 15+15+20

def test_bench_stats():
    bench = DTAHBench()
    stats = bench.stats()
    assert stats["total"] >= 100  # Full set will be 150; currently 130
    assert stats["tier1_count"] == 50

def test_tier_id_parsing():
    assert DTAHBench._tier_from_id("T1-MEP-001") == 1
    assert DTAHBench._tier_from_id("T2-STR-005") == 2
    assert DTAHBench._tier_from_id("T3-HVAC-010") == 3
    assert DTAHBench._tier_from_id("UNKNOWN") is None
