// Export sensitivities for use by external tools (e.g., OpenFFD)

Info<< "Exporting sensitivities for integration with OpenFFD" << nl << endl;

// Create a directory to store sensitivities
if (!runTime.path().exists("sensitivities"))
{
    mkDir(runTime.path()/"sensitivities");
}

// Export sensitivities in OpenFOAM native format
// (they're already written to the time directory)

// Export surface sensitivities for specific patches in a format
// that can be easily read by external tools like OpenFFD

// Get a list of patches to export sensitivities for
wordList patchesToExport;

if (objectiveName == "drag" || objectiveName == "lift" || objectiveName == "moment")
{
    patchesToExport = objDict.get<wordList>("patches");
}
else if (objectiveName == "pressure_loss")
{
    wordList inletPatches = objDict.get<wordList>("inlet");
    wordList outletPatches = objDict.get<wordList>("outlet");
    
    patchesToExport = inletPatches;
    forAll(outletPatches, i)
    {
        patchesToExport.append(outletPatches[i]);
    }
}
else if (objectiveName == "uniformity")
{
    patchesToExport = objDict.get<wordList>("patches");
}

// Export sensitivities for each patch
forAll(patchesToExport, i)
{
    const word& patchName = patchesToExport[i];
    label patchI = mesh.boundaryMesh().findPatchID(patchName);
    
    if (patchI != -1)
    {
        const fvPatch& patch = mesh.boundary()[patchI];
        const vectorField& Cf = patch.Cf();
        
        // Create an export file for this patch
        OFstream exportFile(runTime.path()/"sensitivities"/patchName);
        
        exportFile << "# x y z dx dy dz" << nl;
        
        // Write the face centers and sensitivities
        forAll(patch, faceI)
        {
            const vector& fc = Cf[faceI];
            const vector& sens = boundarySensitivity.boundaryField()[patchI][faceI];
            
            exportFile << fc.x() << " " << fc.y() << " " << fc.z() << " "
                      << sens.x() << " " << sens.y() << " " << sens.z() << nl;
        }
        
        Info<< "Exported " << patch.size() << " sensitivity points for patch: " 
            << patchName << nl << endl;
    }
    else
    {
        Info<< "Warning: Patch " << patchName << " not found, skipping export" << nl << endl;
    }
}

// Export sensitivities in VTK format for visualization
{
    #include "createVTKMesh.H"
    
    // Create a VTK file for the mesh with sensitivities
    vtkMesh.addPointData(meshSensitivity, "sensitivity");
    
    // Also add boundary sensitivity for surfaces
    vtkMesh.addPointData(boundarySensitivity, "boundarySensitivity");
    
    // Write the VTK file
    vtkMesh.write(runTime.path()/"sensitivities"/"sensitivity.vtk");
    
    Info<< "Exported sensitivities in VTK format for visualization" << nl << endl;
}

// Also export in a simple CSV format for easy parsing by external tools
{
    OFstream csvFile(runTime.path()/"sensitivities"/("sensitivity_" + objectiveName + ".csv"));
    
    csvFile << "# CellID,x,y,z,dx,dy,dz" << nl;
    
    forAll(mesh.C(), cellI)
    {
        const vector& cc = mesh.C()[cellI];
        const vector& sens = meshSensitivity[cellI];
        
        csvFile << cellI << ","
               << cc.x() << "," << cc.y() << "," << cc.z() << ","
               << sens.x() << "," << sens.y() << "," << sens.z() << nl;
    }
    
    Info<< "Exported sensitivities in CSV format for all cells" << nl << endl;
}

// Export an OpenFFD-specific format that contains only the surface points and sensitivities
{
    // First collect all boundary faces from the patches of interest
    label totalFaces = 0;
    forAll(patchesToExport, i)
    {
        const word& patchName = patchesToExport[i];
        label patchI = mesh.boundaryMesh().findPatchID(patchName);
        
        if (patchI != -1)
        {
            totalFaces += mesh.boundary()[patchI].size();
        }
    }
    
    // Allocate storage for points and sensitivities
    vectorField points(totalFaces, vector::zero);
    vectorField sensitivities(totalFaces, vector::zero);
    
    // Collect the data
    label pointIndex = 0;
    forAll(patchesToExport, i)
    {
        const word& patchName = patchesToExport[i];
        label patchI = mesh.boundaryMesh().findPatchID(patchName);
        
        if (patchI != -1)
        {
            const fvPatch& patch = mesh.boundary()[patchI];
            const vectorField& Cf = patch.Cf();
            
            forAll(patch, faceI)
            {
                points[pointIndex] = Cf[faceI];
                sensitivities[pointIndex] = boundarySensitivity.boundaryField()[patchI][faceI];
                pointIndex++;
            }
        }
    }
    
    // Export in a format suitable for OpenFFD
    OFstream offdFile(runTime.path()/"sensitivities"/"sensitivity_openffd.dat");
    
    offdFile << "# OpenFFD Sensitivity File" << nl;
    offdFile << "# ObjectiveFunction: " << objectiveName << nl;
    offdFile << "# Points: " << totalFaces << nl;
    offdFile << "# Format: x y z dx dy dz" << nl;
    
    forAll(points, i)
    {
        offdFile << points[i].x() << " " << points[i].y() << " " << points[i].z() << " "
                << sensitivities[i].x() << " " << sensitivities[i].y() << " " << sensitivities[i].z() << nl;
    }
    
    Info<< "Exported " << totalFaces << " surface points and sensitivities in OpenFFD format" << nl << endl;
}

// Create a summary file with the objective value and overall sensitivity statistics
{
    OFstream summaryFile(runTime.path()/"sensitivities"/"summary.txt");
    
    summaryFile << "Objective function: " << objectiveName << nl;
    summaryFile << "Objective value: " << objectiveValue << nl;
    summaryFile << "Sensitivity L2 norm: " << Foam::sqrt(gSum(magSqr(meshSensitivity)) / meshSensitivity.size()) << nl;
    summaryFile << "Maximum sensitivity magnitude: " << gMax(mag(meshSensitivity)) << nl;
    
    vector maxSens = vector::zero;
    vector minSens = vector::zero;
    
    forAll(meshSensitivity, cellI)
    {
        maxSens.x() = max(maxSens.x(), meshSensitivity[cellI].x());
        maxSens.y() = max(maxSens.y(), meshSensitivity[cellI].y());
        maxSens.z() = max(maxSens.z(), meshSensitivity[cellI].z());
        
        minSens.x() = min(minSens.x(), meshSensitivity[cellI].x());
        minSens.y() = min(minSens.y(), meshSensitivity[cellI].y());
        minSens.z() = min(minSens.z(), meshSensitivity[cellI].z());
    }
    
    summaryFile << "Maximum sensitivity components: " << maxSens << nl;
    summaryFile << "Minimum sensitivity components: " << minSens << nl;
    
    Info<< "Created sensitivity summary file" << nl << endl;
}
