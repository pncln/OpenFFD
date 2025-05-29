/*---------------------------------------------------------------------------*\
  Mesh deformation utilities for sensitivity-based shape optimization

  Implements direct mesh deformation based on computed sensitivities,
  with support for various smoothing and regularization techniques.
\*---------------------------------------------------------------------------*/

class MeshDeformer
{
private:
    const Time& runTime_;
    const fvMesh& mesh_;
    const word objectiveName_;
    
    // Deformation parameters
    scalar stepSize_;
    label maxSmoothingIters_;
    scalar smoothingCoeff_;
    bool preserveVolume_;
    bool useRBF_;
    
    // For RBF interpolation
    scalar rbfRadius_;
    word rbfFunction_;
    
public:
    //- Constructor
    MeshDeformer
    (
        const Time& runTime,
        fvMesh& mesh,
        const word& objectiveName
    )
    :
        runTime_(runTime),
        mesh_(mesh),
        objectiveName_(objectiveName),
        stepSize_(0.01),
        maxSmoothingIters_(20),
        smoothingCoeff_(0.5),
        preserveVolume_(true),
        useRBF_(false),
        rbfRadius_(1.0),
        rbfFunction_("wendland")
    {
        // Read configuration from dictionary if available
        IOdictionary deformDict
        (
            IOobject
            (
                "meshDeformDict",
                runTime.constant(),
                mesh,
                IOobject::READ_IF_PRESENT,
                IOobject::NO_WRITE
            )
        );
        
        if (deformDict.headerOk())
        {
            stepSize_ = deformDict.getOrDefault<scalar>("stepSize", 0.01);
            maxSmoothingIters_ = deformDict.getOrDefault<label>("maxSmoothingIters", 20);
            smoothingCoeff_ = deformDict.getOrDefault<scalar>("smoothingCoeff", 0.5);
            preserveVolume_ = deformDict.getOrDefault<bool>("preserveVolume", true);
            useRBF_ = deformDict.getOrDefault<bool>("useRBF", false);
            rbfRadius_ = deformDict.getOrDefault<scalar>("rbfRadius", 1.0);
            rbfFunction_ = deformDict.getOrDefault<word>("rbfFunction", "wendland");
        }
        
        Info<< "Initialized mesh deformation system:" << nl
            << "  Step size: " << stepSize_ << nl
            << "  Smoothing iterations: " << maxSmoothingIters_ << nl
            << "  Smoothing coefficient: " << smoothingCoeff_ << nl
            << "  Preserve volume: " << preserveVolume_ << nl
            << "  Use RBF: " << useRBF_ << nl
            << "  RBF radius: " << rbfRadius_ << nl
            << "  RBF function: " << rbfFunction_ << nl << endl;
    }
    
    //- Destructor
    ~MeshDeformer()
    {}
    
    //- Deform the mesh based on sensitivities
    bool deformMesh
    (
        const volVectorField& sensitivity,
        const scalar& multiplier = 1.0
    )
    {
        Info<< "Deforming mesh based on sensitivities" << nl
            << "  Objective: " << objectiveName_ << nl
            << "  Step multiplier: " << multiplier << nl << endl;
        
        // Get the current mesh points
        pointField oldPoints = mesh_.points();
        pointField newPoints = oldPoints;
        
        // Create displacement field
        vectorField displacement(oldPoints.size(), vector::zero);
        
        // Get boundary mesh
        const polyBoundaryMesh& boundaryMesh = mesh_.boundaryMesh();
        
        // First, set displacement at boundary points based on sensitivities
        if (useRBF_)
        {
            // RBF-based deformation (smoother)
            applyRBFDeformation(sensitivity, displacement, multiplier);
        }
        else
        {
            // Direct sensitivity-based deformation
            applyDirectDeformation(sensitivity, displacement, multiplier);
        }
        
        // Apply smoothing to the displacement field
        smoothDisplacement(displacement);
        
        // Apply volume preservation if requested
        if (preserveVolume_)
        {
            preserveMeshVolume(displacement);
        }
        
        // Apply displacement to points
        forAll(newPoints, pointI)
        {
            newPoints[pointI] += displacement[pointI];
        }
        
        // Update the mesh with new points
        mesh_.movePoints(newPoints);
        
        Info<< "Mesh deformation complete" << nl
            << "  Maximum displacement magnitude: " << gMax(mag(displacement)) << nl
            << "  Average displacement magnitude: " << gAverage(mag(displacement)) << nl << endl;
        
        return true;
    }
    
private:
    //- Apply direct sensitivity-based deformation
    void applyDirectDeformation
    (
        const volVectorField& sensitivity,
        vectorField& displacement,
        const scalar& multiplier
    )
    {
        // Get boundary mesh
        const polyBoundaryMesh& boundaryMesh = mesh_.boundaryMesh();
        
        // Loop over boundary patches
        forAll(boundaryMesh, patchI)
        {
            const polyPatch& patch = boundaryMesh[patchI];
            const word& patchName = patch.name();
            
            // Check if this patch is part of the objective function and should be deformed
            bool deformPatch = false;
            
            // For drag/lift/moment objectives, deform walls
            if ((objectiveName_ == "drag" || objectiveName_ == "lift" || objectiveName_ == "moment") && 
                (patchName.contains("wall") || 
                 patchName.contains("Wing") || 
                 patchName.contains("wing") || 
                 patchName.contains("Airfoil") || 
                 patchName.contains("airfoil")))
            {
                deformPatch = true;
            }
            
            if (deformPatch)
            {
                // Get patch mesh points
                
                // Get patch sensitivities
                const vectorField& patchSens = sensitivity.boundaryField()[patchI];
                
                // Get patch normals for projection
                
                // Face centers for the patch
                
                // Map face sensitivities to points
                Map<vector> pointSensMap;
                Map<label> pointCountMap;
                
                // First, accumulate sensitivities for each point
                forAll(patch, faceI)
                {
                    const face& f = patch[faceI];
                    const vector& faceSens = patchSens[faceI];
                    
                    forAll(f, facePointI)
                    {
                        label pointI = f[facePointI];
                        label meshPointI = patch.meshPoints()[pointI];
                        
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
                
                // Then, compute the average sensitivity for each point
                for (auto iter = pointSensMap.begin(); iter != pointSensMap.end(); ++iter)
                {
                    label meshPointI = iter.key();
                    vector& pointSens = iter.val();
                    label count = pointCountMap[meshPointI];
                    
                    // Average the sensitivity
                    pointSens /= count;
                    
                    // Compute displacement based on sensitivity
                    // Negative sign because we're minimizing the objective
                    vector pointDisp = -stepSize_ * multiplier * pointSens;
                    
                    // Apply the displacement
                    displacement[meshPointI] = pointDisp;
                }
            }
        }
    }
    
    //- Apply RBF-based deformation
    void applyRBFDeformation
    (
        const volVectorField& sensitivity,
        vectorField& displacement,
        const scalar& multiplier
    )
    {
        // This is a simplified placeholder for RBF-based deformation
        // In a full implementation, this would use radial basis functions
        // to interpolate displacement from boundary to volume
        
        // Get boundary mesh
        const polyBoundaryMesh& boundaryMesh = mesh_.boundaryMesh();
        
        // Step 1: Identify boundary points and their displacements
        DynamicList<vector> boundaryPoints;
        DynamicList<vector> boundaryDisplacements;
        DynamicList<label> boundaryPointLabels;
        
        // Loop over boundary patches
        forAll(boundaryMesh, patchI)
        {
            const polyPatch& patch = boundaryMesh[patchI];
            const word& patchName = patch.name();
            
            // Check if this patch is part of the objective function and should be deformed
            bool deformPatch = false;
            
            // For drag/lift/moment objectives, deform walls
            if ((objectiveName_ == "drag" || objectiveName_ == "lift" || objectiveName_ == "moment") && 
                (patchName.contains("wall") || 
                 patchName.contains("Wing") || 
                 patchName.contains("wing") || 
                 patchName.contains("Airfoil") || 
                 patchName.contains("airfoil")))
            {
                deformPatch = true;
            }
            
            if (deformPatch)
            {
                // Get patch mesh points
                
                // Get patch sensitivities
                const vectorField& patchSens = sensitivity.boundaryField()[patchI];
                
                // Face centers for the patch
                
                // Map face sensitivities to points
                Map<vector> pointSensMap;
                Map<label> pointCountMap;
                
                // First, accumulate sensitivities for each point
                forAll(patch, faceI)
                {
                    const face& f = patch[faceI];
                    const vector& faceSens = patchSens[faceI];
                    
                    forAll(f, facePointI)
                    {
                        label pointI = f[facePointI];
                        label meshPointI = patch.meshPoints()[pointI];
                        
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
                
                // Then, compute the average sensitivity for each point
                for (auto iter = pointSensMap.begin(); iter != pointSensMap.end(); ++iter)
                {
                    label meshPointI = iter.key();
                    vector& pointSens = iter.val();
                    label count = pointCountMap[meshPointI];
                    
                    // Average the sensitivity
                    pointSens /= count;
                    
                    // Compute displacement based on sensitivity
                    // Negative sign because we're minimizing the objective
                    vector pointDisp = -stepSize_ * multiplier * pointSens;
                    
                    // Store boundary point and displacement
                    boundaryPoints.append(mesh_.points()[meshPointI]);
                    boundaryDisplacements.append(pointDisp);
                    boundaryPointLabels.append(meshPointI);
                }
            }
        }
        
        // Convert dynamic lists to fields
        vectorField rbfPoints(boundaryPoints.size());
        vectorField rbfDisplacements(boundaryDisplacements.size());
        labelList rbfPointLabels(boundaryPointLabels.size());
        
        forAll(boundaryPoints, i)
        {
            rbfPoints[i] = boundaryPoints[i];
            rbfDisplacements[i] = boundaryDisplacements[i];
            rbfPointLabels[i] = boundaryPointLabels[i];
        }
        
        // Step 2: Apply RBF interpolation for all mesh points
        // This is a simplified version; a real implementation would use
        // proper RBF interpolation functions
        
        // First, apply displacement to boundary points
        forAll(rbfPointLabels, i)
        {
            displacement[rbfPointLabels[i]] = rbfDisplacements[i];
        }
        
        // Then, interpolate to interior points using a simple inverse distance weighting
        forAll(displacement, pointI)
        {
            // Skip boundary points that already have displacement
            if (rbfPointLabels.found(pointI))
            {
                continue;
            }
            
            const point& p = mesh_.points()[pointI];
            scalar totalWeight = 0.0;
            vector interpolatedDisp(0, 0, 0);
            
            // Compute weighted average of displacements
            forAll(rbfPoints, i)
            {
                scalar distance = mag(p - rbfPoints[i]);
                
                // Avoid division by zero
                if (distance < SMALL)
                {
                    interpolatedDisp = rbfDisplacements[i];
                    totalWeight = 1.0;
                    break;
                }
                
                // Use inverse distance weighting
                scalar weight = 1.0 / sqr(distance);
                
                // Apply RBF kernel based on the chosen function
                if (rbfFunction_ == "wendland")
                {
                    // Wendland's C2 function with compact support
                    scalar r = distance / rbfRadius_;
                    if (r < 1.0)
                    {
                        weight = pow(1.0 - r, 4) * (4.0 * r + 1.0);
                    }
                    else
                    {
                        weight = 0.0;
                    }
                }
                else if (rbfFunction_ == "gaussian")
                {
                    // Gaussian RBF
                    weight = Foam::exp(-sqr(distance / rbfRadius_));
                }
                
                interpolatedDisp += weight * rbfDisplacements[i];
                totalWeight += weight;
            }
            
            // Normalize
            if (totalWeight > SMALL)
            {
                interpolatedDisp /= totalWeight;
                displacement[pointI] = interpolatedDisp;
            }
        }
    }
    
    //- Smooth the displacement field
    void smoothDisplacement(vectorField& displacement)
    {
        if (maxSmoothingIters_ <= 0)
        {
            return;
        }
        
        Info<< "Smoothing displacement field with " << maxSmoothingIters_ 
            << " iterations, coefficient " << smoothingCoeff_ << nl << endl;
        
        // Get mesh data for smoothing
        const labelListList& pointPoints = mesh_.pointPoints();
        
        // Iterative Laplacian smoothing
        for (label iter = 0; iter < maxSmoothingIters_; iter++)
        {
            // Store original displacement for this iteration
            vectorField origDisplacement = displacement;
            
            // Apply smoothing
            forAll(displacement, pointI)
            {
                const labelList& neighbors = pointPoints[pointI];
                
                if (neighbors.size() > 0)
                {
                    vector avgDisp(0, 0, 0);
                    
                    forAll(neighbors, nbrI)
                    {
                        avgDisp += origDisplacement[neighbors[nbrI]];
                    }
                    
                    avgDisp /= neighbors.size();
                    
                    // Apply smoothing with weight
                    displacement[pointI] = (1.0 - smoothingCoeff_) * origDisplacement[pointI] + 
                                          smoothingCoeff_ * avgDisp;
                }
            }
        }
    }
    
    //- Preserve mesh volume by adjusting displacement
    void preserveMeshVolume(vectorField& displacement)
    {
        // This is a simplified volume preservation technique
        // In a real implementation, more sophisticated approaches would be used
        
        // Calculate the centroid of displacement
        vector dispCentroid(0, 0, 0);
        scalar totalMag = 0.0;
        
        forAll(displacement, pointI)
        {
            scalar mag = magSqr(displacement[pointI]);
            dispCentroid += mesh_.points()[pointI] * mag;
            totalMag += mag;
        }
        
        if (totalMag > SMALL)
        {
            dispCentroid /= totalMag;
            
            // Calculate volume change due to displacement
            scalar volumeChange = 0.0;
            
            const cellList& cells = mesh_.cells();
            forAll(cells, cellI)
            {
                const cell& c = cells[cellI];
                const labelList& cellPoints = c.labels(mesh_.faces());
                
                // Estimate volume change based on face movement
                forAll(cellPoints, i)
                {
                    label pointI = cellPoints[i];
                    vector dispDir = mesh_.points()[pointI] - dispCentroid;
                    scalar normalDisp = (displacement[pointI] & dispDir) / (mag(dispDir) + SMALL);
                    volumeChange += normalDisp;
                }
            }
            
            // If volume change is significant, apply correction
            if (mag(volumeChange) > SMALL)
            {
                Info<< "Applying volume preservation correction" << nl
                    << "  Estimated volume change: " << volumeChange << nl << endl;
                
                // Scale factor to preserve volume
                scalar scaleFactor = 1.0 - volumeChange / (totalMag + SMALL);
                scaleFactor = Foam::min(Foam::max(scaleFactor, 0.5), 2.0);  // Limit the correction
                
                // Apply correction
                forAll(displacement, pointI)
                {
                    displacement[pointI] *= scaleFactor;
                }
            }
        }
    }
};

// Create mesh deformer
MeshDeformer meshDeformer(runTime, mesh, objectiveName);

// Function to test mesh deformation using computed sensitivities
inline bool testMeshDeformation(const volVectorField& sensitivity, const scalar stepMultiplier = 1.0)
{
    // Read settings from dictionary
    IOdictionary deformDict
    (
        IOobject
        (
            "meshDeformDict",
            runTime.constant(),
            mesh,
            IOobject::READ_IF_PRESENT,
            IOobject::NO_WRITE
        )
    );
    
    bool autoTest = deformDict.getOrDefault<bool>("autoTest", false);
    scalar testStepMultiplier = deformDict.getOrDefault<scalar>("testStepMultiplier", 1.0);
    
    // Override multiplier if provided
    if (stepMultiplier > SMALL)
    {
        testStepMultiplier = stepMultiplier;
    }
    
    if (autoTest)
    {
        Info<< "Performing automatic mesh deformation test" << nl
            << "  Step multiplier: " << testStepMultiplier << nl << endl;
        
        // Save the original mesh points
        pointField origPoints = mesh.points();
        
        // Perform the deformation
        meshDeformer.deformMesh(sensitivity, testStepMultiplier);
        
        // Calculate the objective value with the deformed mesh
        scalar deformedObjective = calculateObjective();
        
        Info<< "Mesh deformation test results:" << nl
            << "  Original objective: " << objectiveValue << nl
            << "  Deformed objective: " << deformedObjective << nl
            << "  Change: " << deformedObjective - objectiveValue << " (" 
            << 100.0 * (deformedObjective - objectiveValue) / objectiveValue << "%)" << nl << endl;
        
        // Restore the original mesh if this is just a test
        if (deformDict.getOrDefault<bool>("restoreAfterTest", true))
        {
            Info<< "Restoring original mesh" << nl << endl;
            mesh.movePoints(origPoints);
        }
        
        return true;
    }
    
    return false;
}
