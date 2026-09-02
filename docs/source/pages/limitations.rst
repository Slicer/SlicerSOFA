Limitations and caveats
=======================

Known caveats
-------------

- **Main-thread stepping.** Simulation steps run on the Qt main thread,
  driven by a zero-interval timer. Heavy scenes therefore make the GUI less
  responsive while the simulation runs; there is no built-in off-thread or
  out-of-process execution.
- **Units and coordinate systems.** Data is passed between MRML and SOFA
  without scaling or axis conversion. Slicer's world coordinate system is RAS
  with millimeters; SOFA scenes are unit-agnostic. Author scenes (material
  parameters, gravity magnitudes, time step) consistently with the units of
  your input data.
- **Plugin set fixed at build time.** Only SOFA modules and plugins compiled
  into the bundled SOFA are available (SofaPython3, STLIB, BeamAdapter,
  Registration, Cosserat, MultiThreading, among others). Scenes requiring
  other plugins will fail to load.
- **Reproducibility of parallel components.** SOFA's multithreaded solvers
  and collision components sum forces in a non-deterministic order, so
  results can vary between identical runs — significantly so for
  marginally-stable scenes. The Sparse Grid Simulation module runs
  single-threaded by default for this reason.
- **MRML→SOFA synchronization is mapping-based.** Only parameter-node fields
  with a registered mapping are synchronized during a running simulation;
  arbitrary edits to other MRML nodes do not propagate into the SOFA scene.
- **SOFA Scene Loader auto-mapping.** Only the component types listed in
  :doc:`sofasceneloader` are automatically mapped to MRML; other components
  simulate normally but are not visualized.
