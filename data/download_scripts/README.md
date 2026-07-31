# Download Scripts

Download scripts and checksums for the four datasets will be added in
**Milestone 1 (Dataset preparation and license verification)** — tracked by
issue #2 *"Prepare dataset download and organization scripts"*.

## Policy (frozen)

1. **Never upload raw data to GitHub.** Only scripts, checksums (e.g., SHA-256),
   and documentation live in this repository.
2. Each dataset directory below will contain:
   - `download.sh` (or `download.py`) — fetches the dataset
   - `sha256sums.txt` — integrity checksums
   - `README.md` — dataset-specific organization notes
3. Download and license status must be confirmed **before** the pilot
   experiment (Milestone 2). If any dataset license restricts academic use,
   the dataset is replaced or dropped and logged in
   [`docs/change_log.md`](../../docs/change_log.md).

## Planned entries

```
download_scripts/
├── ladd/
├── dfire/
├── rescuenet/
└── floodnet/
```
