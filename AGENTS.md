# Notes for agents working on SlicerSOFA

Instructions for any coding agent picking up work here. Conventions
first, then the investigation that is currently open.

## Orientation

SlicerSOFA is a 3D Slicer extension that embeds the SOFA simulation
framework. It is a **SuperBuild**: `SuperBuild/External_Sofa.cmake`
fetches SOFA and five plugins, then configures and builds them as one
external project before the extension itself is built.

```
SuperBuild/External_Sofa.cmake   SOFA + plugins (the long pole)
SlicerSofa/                      the loadable module
SOFASceneLoader/                 scene-loading module
SoftTissueSimulation/            simulation module
SparseGridSimulation/            simulation module
docs/source/pages/               Sphinx docs (reStructuredText)
```

The plugins are pinned by exact SHA in `External_Sofa.cmake`, in this
order: SofaPython3, SofaSTLIB, BeamAdapter, Registration, **Cosserat**.
Order matters when reading logs — Cosserat is configured last, so a
message about Cosserat means SOFA's configure got all the way through
the plugin list.

Building requires a Slicer **build tree** (not an installed Slicer);
`docs/source/pages/building.rst` has the details.

## Working conventions

- Branch off `preview` and open a PR against it. Do not push to
  `preview` directly.
- Commit subjects use the Slicer vocabulary: `ENH:`, `BUG:`, `COMP:`,
  `STYLE:`, `DOC:`, `PERF:`, with an uppercase first word.
- Keep repository text publicly meaningful: no machine hostnames, IP
  addresses, user names or absolute paths from a developer's machine.
  Refer to dashboard builds by id and to changes by PR number.
- Prefer evidence to inference in commit messages. If something was
  measured, give the number; if it was assumed, say so.

## Open: the Windows factory build fails in SOFA's configure step

**Status as of 2026-10-22: unresolved, and the error text is not yet
visible.** This section exists so the next person does not repeat the
elimination already done.

### Symptom

On the Slicer extensions dashboard, the Windows (MSBuild) build fails
while Linux and macOS pass. Example: build `4463171`, 2026-10-21.

CDash attributes it as a *compilation* error, which is misleading. The
MSBuild failure is `MSB8066` on `Sofa-configure.rule`, i.e. the **inner
SOFA external project's configure step**. The outer SlicerSOFA configure
is clean on all three platforms.

```
CMake Error at .../Sofa-stamp/Sofa-configure-Release.cmake:47 (message):
  Stopping after outputting logs.
```

### What is established

- It began with the SOFA **v26.06** update (`937978b`). `ce24f33`
  records that no Windows package has been produced since r34908
  (2026-08-22, still v25.12).
- The tinyxml2 fix in `ce24f33` **worked**. The original failure was an
  early `Cannot find TinyXML2`; configure now proceeds through all five
  plugins and dies after Cosserat, the last one. This is therefore a
  *second*, different failure behind the first.
- The retained log tail ends on an ordinary
  `-- SofaPython3 bindings for Cosserat will be created.` with no CMake
  error near it, and no `Configuring incomplete, errors occurred!`.

### Why the error is invisible, and what was done about it

`LOG_CONFIGURE` keeps stdout and stderr in **separate** files. On
failure, ExternalProject prints the last **10 KB of each** — the limit
is hard-coded in its `read_up_to_max_size` helper, so there is no knob —
and CDash retains only a small window of that around the error. Two
truncations stack, and because the streams are split, the surviving tail
is the chatty stream (stdout progress) rather than the informative one
(stderr, where CMake errors go).

`LOG_MERGED_STDOUTERR 1` was added for exactly this reason: merged, the
retained tail is chronological, so it should contain whatever actually
preceded the failure.

**First step for whoever picks this up: look at a dashboard build newer
than that change and read the merged tail.** The error may simply be
there now.

### If the log still does not say

Reproduce locally. This needs a Windows machine with Visual Studio 2022,
CMake ≥ 3.31, Git, Python 3.12, Qt 5.15 (the Windows factory builds with
Qt 5.15, unlike Linux/macOS), and a Slicer build tree.

Two Windows-specific traps worth setting up before blaming the code:

- **Path length.** The factory uses a deliberately short build root to
  stay under `MAX_PATH`. Use a short root such as `C:\S`, and enable
  long paths (`LongPathsEnabled` in
  `HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem`, plus
  `git config --system core.longpaths true`). A path-length failure in a
  deeply nested plugin tree is easy to mistake for the real bug.
- **Antivirus.** Real-time scanning inspects every one of the millions
  of small files a SuperBuild writes. Excluding the build and source
  roots materially changes build time and can change failure modes.

### Hypotheses not yet tested

- The configure process may be terminating **abnormally** rather than
  reporting a CMake error. The absence of any stderr content and of
  `Configuring incomplete, errors occurred!` in the retained tail is
  consistent with a crash or a kill, not with a diagnosed failure. The
  merged log should distinguish these.
- Cosserat's `Binding/CMakeLists.txt` emits a CMake *developer warning*
  at the point where the log ends. A warning is not the failure, but it
  is the last thing SOFA's configure reported.

Do not conclude from the Cosserat messages alone that Cosserat is at
fault: it is simply the last plugin processed, so anything failing after
the plugin loop would look the same.
