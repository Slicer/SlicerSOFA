# Notes for agents working on SlicerSOFA

## What this repository is

A 3D Slicer extension embedding the SOFA simulation framework. It is a
**SuperBuild**: `SuperBuild/External_Sofa.cmake` fetches SOFA and its
plugins, then configures and builds them as one external project before
the extension itself is built.

```
SuperBuild/External_*.cmake   SOFA and its dependencies (the long pole)
SlicerSofa/                   the loadable module
SOFASceneLoader/              scene-loading module
SoftTissueSimulation/         simulation module
SparseGridSimulation/         simulation module
docs/source/pages/            Sphinx documentation (reStructuredText)
```

SOFA and every plugin are pinned by exact SHA in `External_Sofa.cmake`.
Bump those pins deliberately, never incidentally: a version bump is the
most likely cause of a build that broke without anyone touching this
repository.

Building requires a Slicer **build tree**, not an installed Slicer. See
`docs/source/pages/building.rst`.

## Conventions

- Branch off `preview` and open a PR against it. Never push to `preview`
  directly.
- Commit subjects use the Slicer vocabulary — `ENH:`, `BUG:`, `COMP:`,
  `STYLE:`, `DOC:`, `PERF:` — with an uppercase first word.
- Keep repository text publicly meaningful: no machine hostnames, IP
  addresses, user names or absolute paths from a developer's machine.
  Cite dashboard builds by id and changes by commit or PR number.
- Prefer evidence to inference. If something was measured, give the
  number; if it was assumed, say so. A rejected hypothesis is worth
  recording — it stops the next person repeating it.

## Working with the dashboard

Extension builds appear on the Slicer extensions dashboard. Two things
about its output mislead reliably:

**A failing SuperBuild step is reported as a COMPILATION error.** When
an inner external project fails, MSBuild raises `MSB8066` on that
project's `*.rule`, and CDash counts it under compilation. The outer
SlicerSOFA configure can be clean on all platforms while an inner
project's *configure* is what actually failed. Read the step name in the
error, not the counter.

**The retained log is heavily truncated, twice.** `LOG_CONFIGURE` keeps
stdout and stderr in separate files; on failure ExternalProject prints
the last 10 KB of each (hard-coded in its `read_up_to_max_size` helper),
and CDash retains only a window of that. `LOG_MERGED_STDOUTERR 1`
interleaves the streams so the surviving tail is chronological rather
than whichever stream was chattier.

Consequently, the last line before a failure is usually *not* the cause.
CMake flushes deferred `CMake Error in ...` blocks at the very end, long
after the `--` progress line that appears to precede them. Whichever
component is configured last will look guilty and almost never is.

## Building and debugging on Windows

Windows is where this project breaks, and almost always at configure
time rather than compile time.

Requirements: Visual Studio 2022, CMake >= 3.31, Git, Python 3.12, and
Qt 5.15 — the Windows factory builds with Qt 5.15 even though Linux and
macOS use Qt 6.

Reaching SOFA's configure from a clean build directory takes roughly 15
minutes, so a local reproduction turns a nightly cycle into a loop of
minutes. It is almost always worth doing.

The full, untruncated logs are on disk, which is the advantage over the
dashboard:

```
<build>/Sofa-prefix/src/Sofa-stamp/Sofa-configure-out.log
<build>/Sofa-prefix/src/Sofa-stamp/Sofa-configure-err.log
```

To force the configure step to re-run, delete **only** the stamp file
`Sofa-configure` and `<build>/Sofa-build/CMakeCache.txt`, then re-run
the outer cmake first. Do not wildcard-delete `Sofa-configure*`: that
removes CMake's generated per-config driver scripts, and the build then
fails with `Not a file: ...Sofa-configure-Release.cmake`, which looks
like a real failure and is not.

Two environment traps manufacture failures that resemble real ones:

- **Path length.** Use a short build root (the factory does, to stay
  under `MAX_PATH`), and enable long paths — `LongPathsEnabled` under
  `HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem`, plus
  `git config --system core.longpaths true`.
- **Antivirus.** Real-time scanning inspects every one of the millions
  of small files a SuperBuild writes. Excluding the build and source
  roots changes build times materially and can change failure modes.

### Imported targets from Slicer's third-party libraries

A recurring class of Windows-only failure, worth recognising on sight.

Slicer provides several third-party libraries **static, with no DLL**.
On Windows, `Find<Pkg>` modules see a `.lib`, assume it is an import
library for a DLL, and create the imported target as **SHARED**:
`IMPORTED_IMPLIB` points at the `.lib` while `IMPORTED_LOCATION` — which
for a shared imported target means the DLL — is left empty. Anything
that then reads `LOCATION_<CONFIG>`, as SOFA's install macros do, fails.

Linux is immune: with no import-library concept the target is `UNKNOWN`
with `IMPORTED_LOCATION` on the `.a`. The discriminator is the imported
target **TYPE**, not static-vs-shared — the libraries are static on both
platforms.

Two instances are in the history: `tinyxml2` (fixed by building it
shared) and `zlib` (fixed by declaring the target as `STATIC IMPORTED`
before SOFA looks for it, injected via `CMAKE_PROJECT_Sofa_INCLUDE`).
Prefer the second shape — a second copy of a library Slicer already
provides risks two copies in one process, and patches nothing upstream.
`<Pkg>_USE_STATIC_LIBS` does not help when the library path is already
in the cache, because the find module short-circuits first.

## Diagnosis practice

Two habits earn their cost on this codebase:

**Instrument rather than infer.** When a CMake target misbehaves, print
its properties at the failing call and compare against a target that
works. A contrast between a broken and a working case identifies the
differing property immediately, where reasoning from the error message
tends to produce plausible wrong answers.

**Check that the thing you are testing actually ran.** An absent error
and an absent test look identical in a log. Assert on a positive signal
— a marker message, a non-empty log, a written stamp — before trusting a
green result.
