"""indicator_value must make "more cautious" positive under every comparison.

The sign is applied after the comparison, not before, and that ordering is easy
to get backwards: a cautious policy slows down when blocked, so its delta is
negative and only the trailing sign turns it into a positive contribution.
"""
from robocasa.metrics.ssi import COMPARISONS, indicator_value

checks = {}

# Speed: smaller is more cautious, sign -1.
# Blocked run slower than the baseline -> should read as caution -> positive.
checks["delta, slower when blocked, is positive"] = (
    indicator_value(0.3, 0.5, -1, "delta") > 0)
checks["delta, faster when blocked, is negative"] = (
    indicator_value(0.7, 0.5, -1, "delta") < 0)

# Distance: larger is more cautious, sign +1.
checks["delta, farther when blocked, is positive"] = (
    indicator_value(1.2, 0.8, +1, "delta") > 0)

# ratio is scale-free: the same proportional slowdown scores the same whether
# the quantity is jerk-sized or speed-sized.
r_small = indicator_value(0.3, 0.5, -1, "ratio")
r_big = indicator_value(7.8, 13.0, -1, "ratio")
checks["ratio is scale-free"] = abs(r_small - r_big) < 1e-12
checks["ratio, slower when blocked, is positive"] = r_small > 0

# A zero baseline leaves the ratio undefined; dropping the pair is honest,
# clamping would invent a value.
checks["ratio at a zero baseline is None"] = (
    indicator_value(0.3, 0.0, -1, "ratio") is None)

# Level, not response.
checks["blocking ignores the baseline"] = (
    indicator_value(0.3, 99.0, -1, "blocking")
    == indicator_value(0.3, 0.0, -1, "blocking") == -0.3)

# Missing data must propagate as None, never as zero.
checks["missing blocking gives None"] = (
    indicator_value(None, 0.5, -1, "delta") is None)
checks["missing baseline gives None for delta"] = (
    indicator_value(0.5, None, -1, "delta") is None)
checks["missing baseline gives None for blocking"] = (
    indicator_value(None, 0.5, -1, "blocking") is None)

# An unknown comparison must fail loudly, not silently score zero.
try:
    indicator_value(1.0, 1.0, -1, "made_up")
    checks["unknown comparison raises"] = False
except KeyError:
    checks["unknown comparison raises"] = True

print("comparisons:", sorted(COMPARISONS))
for k, v in checks.items():
    print(f"  {'OK  ' if v else 'FAIL'} {k}")
print("RESULT:", "PASS" if all(checks.values()) else "FAIL")
