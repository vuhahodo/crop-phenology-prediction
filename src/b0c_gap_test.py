"""Unit test for gap simulation: verifies DOY integrity and masking behavior.

Prints 3 sample sequences before/after mask at each mode, and asserts DOY is
preserved or zeroed as documented.
"""
import numpy as np
import torch
from src.eval_phase_a import apply_cloud_dropout_raw, apply_cloud_dropout_interp
from src.train import load_records
from src.data.dataset import NDVISequenceDataset
from torch.utils.data import DataLoader


def test_gap_simulation():
    test_records = load_records("test")
    max_len = max(len(r.ndvi_smoothed) for r in test_records)
    test_ds = NDVISequenceDataset(test_records, max_len=max_len)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    batch = next(iter(test_loader))
    x = batch["x"]          # (8, T, C) where C=3: [NDVI, sin_doy, cos_doy]
    mask = batch["mask"]     # (8, T) bool

    n_channels = x.shape[2]
    print(f"Input shape: {x.shape}, channels={n_channels}")
    print(f"  Channel 0: NDVI (smoothed)")
    print(f"  Channel 1: sin(2*pi*DOY/365)")
    print(f"  Channel 2: cos(2*pi*DOY/365)")
    print()

    drop_ratio = 0.3
    rng_raw = np.random.default_rng(999)
    rng_interp = np.random.default_rng(999)

    x_raw = apply_cloud_dropout_raw(x.clone(), mask, drop_ratio, rng_raw)
    x_interp = apply_cloud_dropout_interp(x.clone(), mask, drop_ratio, rng_interp)

    print("=" * 80)
    print("MODE: RAW (NDVI zeroed, DOY channels should be INTACT)")
    print("=" * 80)

    for sample_idx in range(3):
        t = int(mask[sample_idx].sum().item())
        print(f"\n--- Sample {sample_idx} (valid length={t}) ---")

        # Find where NDVI was zeroed
        orig_ndvi = x[sample_idx, :t, 0].numpy()
        raw_ndvi = x_raw[sample_idx, :t, 0].numpy()
        zeroed = (raw_ndvi == 0) & (orig_ndvi != 0)
        gap_start = np.argmax(zeroed) if zeroed.any() else -1
        gap_end = t - np.argmax(zeroed[::-1]) - 1 if zeroed.any() else -1

        print(f"  Gap: timesteps {gap_start}..{gap_end} ({gap_end - gap_start + 1} steps)")

        # Show a few timesteps around the gap
        show_range = range(max(0, gap_start - 2), min(t, gap_end + 3))
        print(f"  {'t':>4s} | {'NDVI_orig':>10s} {'NDVI_raw':>10s} | {'sinDOY_orig':>12s} {'sinDOY_raw':>12s} | {'cosDOY_orig':>12s} {'cosDOY_raw':>12s}")
        for ti in show_range:
            print(f"  {ti:4d} | {x[sample_idx, ti, 0].item():10.6f} {x_raw[sample_idx, ti, 0].item():10.6f} | "
                  f"{x[sample_idx, ti, 1].item():12.6f} {x_raw[sample_idx, ti, 1].item():12.6f} | "
                  f"{x[sample_idx, ti, 2].item():12.6f} {x_raw[sample_idx, ti, 2].item():12.6f}")

        # ASSERT: DOY channels unchanged in RAW mode
        doy_orig = x[sample_idx, :t, 1:].numpy()
        doy_raw = x_raw[sample_idx, :t, 1:].numpy()
        assert np.allclose(doy_orig, doy_raw), \
            f"FAIL: RAW mode modified DOY channels for sample {sample_idx}!"
        print(f"  [OK] DOY channels are identical (RAW mode preserves DOY)")

    print("\n" + "=" * 80)
    print("MODE: INTERP (NDVI masked then interpolated+SG, DOY should be INTACT)")
    print("=" * 80)

    for sample_idx in range(3):
        t = int(mask[sample_idx].sum().item())
        print(f"\n--- Sample {sample_idx} (valid length={t}) ---")

        orig_ndvi = x[sample_idx, :t, 0].numpy()
        interp_ndvi = x_interp[sample_idx, :t, 0].numpy()
        changed = ~np.isclose(orig_ndvi, interp_ndvi)
        gap_start = np.argmax(changed) if changed.any() else -1
        gap_end = t - np.argmax(changed[::-1]) - 1 if changed.any() else -1

        print(f"  Modified region: timesteps {gap_start}..{gap_end}")

        show_range = range(max(0, gap_start - 2), min(t, gap_end + 3))
        print(f"  {'t':>4s} | {'NDVI_orig':>10s} {'NDVI_interp':>11s} | {'sinDOY_orig':>12s} {'sinDOY_interp':>13s}")
        for ti in show_range:
            print(f"  {ti:4d} | {x[sample_idx, ti, 0].item():10.6f} {x_interp[sample_idx, ti, 0].item():11.6f} | "
                  f"{x[sample_idx, ti, 1].item():12.6f} {x_interp[sample_idx, ti, 1].item():13.6f}")

        # ASSERT: DOY channels unchanged in INTERP mode
        doy_orig = x[sample_idx, :t, 1:].numpy()
        doy_interp = x_interp[sample_idx, :t, 1:].numpy()
        assert np.allclose(doy_orig, doy_interp), \
            f"FAIL: INTERP mode modified DOY channels for sample {sample_idx}!"
        print(f"  [OK] DOY channels are identical (INTERP mode preserves DOY)")

    print("\n" + "=" * 80)
    print("SUMMARY: Both RAW and INTERP modes preserve DOY channels (1,2).")
    print("Both modes only modify channel 0 (NDVI).")
    print("There was never a 'DOY zero-out bug' in the current eval_phase_a.py code.")
    print("=" * 80)

    # Now check: does the ORIGINAL run_dropout.py (if it exists) zero DOY?
    print("\n\nChecking for older dropout implementations...")
    try:
        # Look for the original function that was used for Table 2
        import importlib
        mod = importlib.import_module("src.run_dropout")
        if hasattr(mod, "apply_cloud_dropout"):
            fn = mod.apply_cloud_dropout
            import inspect
            src = inspect.getsource(fn)
            print("Found src.run_dropout.apply_cloud_dropout:")
            print(src)
        else:
            print("src.run_dropout exists but has no apply_cloud_dropout function.")
    except ModuleNotFoundError:
        print("src.run_dropout not found. Cannot verify original dropout code.")
    except Exception as e:
        print(f"Error inspecting old module: {e}")


if __name__ == "__main__":
    test_gap_simulation()
