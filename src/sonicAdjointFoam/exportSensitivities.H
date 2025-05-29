// Export sensitivities in a format compatible with OpenFFD
Info<< "Exporting sensitivities for OpenFFD integration..." << endl;

// Create output directory
fileName outputDir = runTime.path()/"sensitivities";
mkDir(outputDir);

// Determine which patches to export sensitivities for
labelList exportPatchIDs;

// If specific patches are defined for export, use those
if (optimizationDict.found("exportPatches"))
{
    wordList exportPatchNames = optimizationDict.lookup("exportPatches");
    forAll(exportPatchNames, i)
    {
        label patchID = mesh.boundaryMesh().findPatchID(exportPatchNames[i]);
        if (patchID >= 0)
        {
            exportPatchIDs.append(patchID);
        }
        else
        {
            WarningInFunction
                << "Patch " << exportPatchNames[i] << " not found." << endl;
        }
    }
}
else
{
    // Otherwise, use the same patches as the objective function
    exportPatchIDs = objPatchIDs;
}

if (exportPatchIDs.size() == 0)
{
    FatalErrorInFunction
        << "No patches specified for sensitivity export."
        << exit(FatalError);
}

// Choose export format based on setting
if (sensFormat == "OpenFFD" || sensFormat == "CSV")
{
    // Export surface mesh points and sensitivities in CSV format for OpenFFD
    OFstream sensFile(outputDir/"surface_sensitivities.csv");
    
    sensFile << "# Surface sensitivity data for OpenFFD" << nl
             << "# Format: x,y,z,sensitivity_x,sensitivity_y,sensitivity_z,patch_id,patch_name" << nl;
    
    // Export surface mesh points and sensitivities
    forAll(exportPatchIDs, i)
    {
        const label patchID = exportPatchIDs[i];
        const word& patchName = mesh.boundaryMesh()[patchID].name();
        
        // Get patch information
        const polyPatch& pp = mesh.boundaryMesh()[patchID];
        
        // Get mesh points for this patch
        const labelList& meshPoints = pp.meshPoints();
        
        // Get patch face centers
        const vectorField& faceCenters = mesh.Cf().boundaryField()[patchID];
        
        // Get sensitivities on this patch
        const vectorField& patchSens = meshSensitivity.boundaryField()[patchID];
        
        // Create point-to-face mapping to transfer face-based sensitivities to points
        // This is a simplified approach; a more sophisticated method would weight by area
        Map<vector> pointSensMap;
        Map<label> pointCountMap;
        
        // First, accumulate sensitivities for each point
        forAll(pp, faceI)
        {
            const face& f = pp[faceI];
            const vector& faceSens = patchSens[faceI];
            
            forAll(f, facePointI)
            {
                label pointI = f[facePointI];
                label meshPointI = pp.meshPoints()[pointI];
                
                if (!pointSensMap.found(meshPointI))
                {
                    pointSensMap.insert(meshPointI, faceSens);
                    pointCountMap.insert(meshPointI, 1);
                }
                else
                {
                    pointSensMap[meshPointI] += faceSens;
                    pointCountMap[meshPointI]++;
                }
            }
        }
        
        // Export sensitivities for each point
        forAll(meshPoints, pointI)
        {
            label meshPointI = meshPoints[pointI];
            const point& pt = mesh.points()[meshPointI];
            
            vector pointSens = vector::zero;
            if (pointSensMap.found(meshPointI))
            {
                pointSens = pointSensMap[meshPointI] / pointCountMap[meshPointI];
            }
            
            // Write CSV line: x,y,z,sens_x,sens_y,sens_z,patch_id,patch_name
            sensFile << pt.x() << "," << pt.y() << "," << pt.z() << ","
                     << pointSens.x() << "," << pointSens.y() << "," << pointSens.z() << ","
                     << patchID << "," << patchName << nl;
        }
    }
    
    Info<< "Exported sensitivity data to: " << sensFile.name() << nl << endl;
    
    // Also export a metadata file with objective function information
    OFstream metaFile(outputDir/"sensitivity_metadata.json");
    
    metaFile << "{" << nl
             << "  \"objectiveType\": \"" << objectiveType << "\"," << nl
             << "  \"objectiveValue\": " << objectiveValue << "," << nl
             << "  \"stepSize\": " << stepSize << "," << nl
             << "  \"normalizedSensitivity\": true," << nl
             << "  \"smoothed\": " << smoothSensitivity << nl
             << "}" << nl;
    
    Info<< "Exported metadata to: " << metaFile.name() << nl << endl;
}
else if (sensFormat == "VTK")
{
    // Alternative: Export in VTK format for visualization
    #include "exportVTK.H"
}
else
{
    FatalErrorInFunction
        << "Unknown sensitivity export format: " << sensFormat << nl
        << "Valid options are: OpenFFD, CSV, VTK" << nl
        << exit(FatalError);
}

// Success message
Info<< "Sensitivity export completed successfully" << nl 
    << "These files can now be used with OpenFFD for shape optimization" << nl << endl;
