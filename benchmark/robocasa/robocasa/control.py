"""Control-clock constants shared by environments and run-time logging."""

# One 20 Hz control-step clock for environment trajectories and SSI ledgers.
# Do not introduce layer-specific logging intervals: they would make the
# persisted trajectory a downsampled version of the dynamics it is scored on.
CONTROL_LOG_INTERVAL_STEPS = 1
