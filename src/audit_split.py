"""Data audit: check for parcel leakage between train/val/test splits."""
from src.train import load_records


def main():
    train = load_records("train")
    val = load_records("val")
    test = load_records("test")

    train_ids = set(r.season_id for r in train)
    val_ids = set(r.season_id for r in val)
    test_ids = set(r.season_id for r in test)

    print(f"Train: {len(train_ids)} unique season_ids")
    print(f"Val:   {len(val_ids)} unique season_ids")
    print(f"Test:  {len(test_ids)} unique season_ids")
    print(f"Total: {len(train_ids | val_ids | test_ids)}")
    print()

    # Check overlaps
    tv = train_ids & val_ids
    tt = train_ids & test_ids
    vt = val_ids & test_ids
    print(f"Train & Val overlap:  {len(tv)}")
    print(f"Train & Test overlap: {len(tt)}")
    print(f"Val & Test overlap:   {len(vt)}")
    print()

    # Extract parcel IDs (field IDs) from season_ids
    # Format: "frh01_{fid}_{classname}"
    def extract_fid(sid):
        parts = sid.split("_")
        # frh01_XXXX_classname -> fid is parts[1]
        return parts[1] if len(parts) >= 3 else sid

    train_fids = set(extract_fid(r.season_id) for r in train)
    val_fids = set(extract_fid(r.season_id) for r in val)
    test_fids = set(extract_fid(r.season_id) for r in test)

    print(f"Train field IDs: {len(train_fids)}")
    print(f"Val field IDs:   {len(val_fids)}")
    print(f"Test field IDs:  {len(test_fids)}")

    fid_tv = train_fids & val_fids
    fid_tt = train_fids & test_fids
    fid_vt = val_fids & test_fids
    print(f"Train & Val field overlap:  {len(fid_tv)}")
    print(f"Train & Test field overlap: {len(fid_tt)}")
    print(f"Val & Test field overlap:   {len(fid_vt)}")
    print()

    # Show some sample IDs
    print("Sample train IDs:", [r.season_id for r in train[:5]])
    print("Sample test IDs:", [r.season_id for r in test[:5]])

    # BreizhCrops is single-year (2017), so each parcel appears once.
    # But verify: are there any duplicate fids within a single split?
    all_fids = [extract_fid(r.season_id) for r in train + val + test]
    from collections import Counter
    c = Counter(all_fids)
    dupes = {fid: cnt for fid, cnt in c.items() if cnt > 1}
    print(f"\nField IDs appearing >1 time across all splits: {len(dupes)}")
    if dupes:
        for fid, cnt in list(dupes.items())[:5]:
            print(f"  fid={fid} appears {cnt} times")


if __name__ == "__main__":
    main()
