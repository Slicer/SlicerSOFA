Limitations and Caveats
=======================

Known caveats
-------------

- Unit conventions: SOFA typically uses meters (SI) while Slicer often uses millimeters in some pipelines. SlicerSOFA tries to keep units consistent but users should ensure correct scaling when importing/exporting geometry.
- Mesh formats: Slicer can import OBJ and STL; if a SOFA scene contains other formats you may need to convert them prior to loading.
- Threading: The integration currently executes simulation steps on the main thread via a Qt timer. Long-running simulations can block the GUI; for heavy simulations consider an out-of-process solver or adopt a custom multithreaded approach with careful MRML/VTK synchronization.
- SOFA plugins: Only SOFA features and plugins available at build-time will be supported. If your scene relies on third-party SOFA plugins, add them to the SuperBuild or provide them at runtime.
- Persistence: Saving and restoring scenes that include live SOFA scenes is supported via MRML nodes but complex scenes with external file paths may require path rewrites.

Missing features (future work)
------------------------------

- Full two-way editing (editing Slicer model -> propagate to SOFA scene) is limited.
- Advanced UI for complex SOFA parameter editing is not yet complete.
- Integration with remote/hardware-in-the-loop simulation backends is not included out-of-the-box.
