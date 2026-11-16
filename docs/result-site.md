# Result visualization site

The site displays the runner's recorded results. It does not run benchmarks or score tensor
outputs during page requests. The existing HTML tables and SVG performance charts are shared
by the local server and the single-file, read-only export.

## Open a run or a collection

```bash
# Any directory containing run bundles, including nested collections
python scripts/manager.py --dir runs --port 6688

# The validated eight-prompt, two-backend run
python scripts/manager.py \
  --dir runs/runner-v2-validation/20260908T190954Z-e1896bde --port 6688

# Standalone snapshot: no server, external assets, or write actions
python scripts/manager.py --dir runs/runner-v2-validation --export results.html
```

The server listens only on `127.0.0.1`. Open `http://127.0.0.1:6688`.

Each run bundle supplies `plan.json`, frozen `experiment.json` definitions, job `execution.json`
and `result.json` records, and the scorer's `report.json`. The viewer checks manifest identities
and cell coverage, then displays the saved comparisons. It checks that completed jobs have a
tensor artifact but does not open or hash its contents. This is a report viewer, not an integrity
re-audit: use `scripts/bench.py score RUN_DIRECTORY` to validate artifacts and regenerate reports.

Missing reports show execution states (`RAN`, `ERROR`, etc.) with a warning, never an invented
correctness verdict. Failed or incomplete jobs remain visible. Invalid manifests produce warnings
without hiding other valid runs.

The overview explicitly aggregates recorded history. Method matrices retain one row per recorded
experiment, not a guessed "latest" result. Job pages show backend directory names, image ID,
experiment/input hashes, source identity, comparison mode, and the explicit reference job.
Procedure snippets are from the current checkout; they are not historical source snapshots.

## Inbox, archive, and trash

```bash
python scripts/manager.py --dir runs/corpus --inbox runs/inbox
```

The inbox lists immediate child run bundles and `.pt` run files. Archive moves the
entire bundle, including logs, frozen inputs, reports, and tensor outputs. An existing destination
is never overwritten; the action returns a conflict instead. Select a collection as the archive
destination, not an existing run bundle.

"Move to trash" moves a run into `INBOX/.trash/UNIQUE_ID/RUN_NAME`. Nothing is permanently
deleted. Recover a run by moving it back into an empty destination with the same name. Hidden
trash directories are excluded from discovery.

Moves require Linux `renameat2` support and source/destination on the same filesystem. Unsupported
or cross-filesystem moves fail without copying/deleting artifacts. POST actions require a token
from the inbox page, and the server rejects unexpected hosts and cross-origin actions. Static
exports contain no action forms.

## Explicit run-file import

The construct probes (`scripts/constructs.py`) and the perf microbenchmark (`scripts/perf.py`) write
`.pt` run files with version-4 coordinates, which need a one-time import. The import also reads
version 3; older coordinate formats must be rerun:

```bash
python scripts/manager.py --dir runs/probes --import-run-files --export probes.html
```

The import loads tensors and plain containers only (`torch.load(weights_only=True)`). It saves
small `NAME.summary.json` sidecars so subsequent browsing reads JSON only. A run file holding
neither probe results nor performance rows is skipped; method runs are job bundles of
`bench.py run`.

Changing a `.pt` file's size or modification time invalidates its summary; rerun the import.
Archive and trash move its summary alongside the artifact. No tensor cache is kept by the site.
