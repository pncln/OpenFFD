// Export sensitivity data in VTK format for visualization
Info<< "Exporting sensitivities in VTK format..." << endl;

// Create VTK directory
fileName vtkDir = outputDir/"vtk";
mkDir(vtkDir);

// Export boundary mesh with sensitivities
forAll(exportPatchIDs, i)
{
    const label patchID = exportPatchIDs[i];
    const word& patchName = mesh.boundaryMesh()[patchID].name();
    
    // Get patch information
    const polyPatch& pp = mesh.boundaryMesh()[patchID];
    
    // Create VTK file name
    fileName vtkFileName = vtkDir/patchName + "_sensitivity.vtk";
    
    // Open VTK file
    OFstream vtkFile(vtkFileName);
    
    // Write VTK header
    vtkFile << "# vtk DataFile Version 2.0" << nl
            << "Surface sensitivities for patch " << patchName << nl
            << "ASCII" << nl
            << "DATASET POLYDATA" << nl;
    
    // Write points
    const pointField& points = mesh.points();
    const labelList& meshPoints = pp.meshPoints();
    
    vtkFile << "POINTS " << meshPoints.size() << " float" << nl;
    
    forAll(meshPoints, pointI)
    {
        const point& pt = points[meshPoints[pointI]];
        vtkFile << pt.x() << " " << pt.y() << " " << pt.z() << nl;
    }
    
    // Write polygons (faces)
    label nFaces = pp.size();
    label totalSize = 0;
    
    forAll(pp, faceI)
    {
        const face& f = pp[faceI];
        totalSize += 1 + f.size();
    }
    
    vtkFile << "POLYGONS " << nFaces << " " << totalSize << nl;
    
    // Create map from mesh points to local points
    Map<label> meshPointMap;
    forAll(meshPoints, i)
    {
        meshPointMap.insert(meshPoints[i], i);
    }
    
    // Write faces
    forAll(pp, faceI)
    {
        const face& f = pp[faceI];
        vtkFile << f.size();
        
        forAll(f, fp)
        {
            label localIdx = meshPointMap[pp.meshPoints()[f[fp]]];
            vtkFile << " " << localIdx;
        }
        vtkFile << nl;
    }
    
    // Write sensitivity data as point data
    vtkFile << "POINT_DATA " << meshPoints.size() << nl;
    vtkFile << "VECTORS sensitivity float" << nl;
    
    // Get sensitivities on this patch
    const vectorField& patchSens = meshSensitivity.boundaryField()[patchID];
    
    // Create point-to-face mapping to transfer face-based sensitivities to points
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
    
    // Write point sensitivities
    forAll(meshPoints, pointI)
    {
        label meshPointI = meshPoints[pointI];
        
        vector pointSens = vector::zero;
        if (pointSensMap.found(meshPointI))
        {
            pointSens = pointSensMap[meshPointI] / pointCountMap[meshPointI];
        }
        
        vtkFile << pointSens.x() << " " << pointSens.y() << " " << pointSens.z() << nl;
    }
    
    Info<< "Exported VTK sensitivity data for patch " << patchName 
        << " to: " << vtkFileName << nl;
}

Info<< "VTK export completed." << nl << endl;
