/*---------------------------------------------------------------------------*\
  Setup for exporting results

  Sets up directories and utilities for exporting sensitivities and other fields for
  visualization and post-processing.
\*---------------------------------------------------------------------------*/

// Create directory for output
fileName exportDir = runTime.path()/"sensitivities";
if (!isDir(exportDir))
{
    mkDir(exportDir);
}

Info<< "Created export directory for sensitivities" << nl << endl;

// We'll use standard OpenFOAM file writers instead of VTK
// This avoids the dependency on VTK libraries
