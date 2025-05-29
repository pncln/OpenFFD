/*---------------------------------------------------------------------------*\
  Adjoint verification tools

  Implements verification methods for the adjoint solver:
  1. Identity test - checks if (dR/dU)^T * psi = dJ/dU
  2. Finite difference verification for sensitivity calculations
  3. Compares adjoint-based sensitivities with finite differences
\*---------------------------------------------------------------------------*/

Info<< "Verifying adjoint implementation" << nl << endl;

// Create a verification directory
if (!runTime.path().exists("verification"))
{
    mkDir(runTime.path()/"verification");
}

// ===== Identity Test =====
// This test verifies that the adjoint operator A^T is correct
// If psi is the adjoint solution and v is any vector, then
// (A^T * psi, v) = (psi, A * v) = (dJ/dU, v)

// Only perform identity test if specified in adjointDict
bool performIdentityTest = adjointDict.getOrDefault<bool>("performIdentityTest", false);

if (performIdentityTest)
{
    Info<< "Performing adjoint identity test..." << nl << endl;
    
    // Create test vectors
    volScalarField testRho
    (
        IOobject
        (
            "testRho",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("testRho", dimDensity, 1.0)
    );
    
    volVectorField testU
    (
        IOobject
        (
            "testU",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedVector("testU", dimVelocity, vector(1, 0, 0))
    );
    
    volScalarField testE
    (
        IOobject
        (
            "testE",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("testE", dimEnergy/dimMass, 1.0)
    );
    
    // Initialize test vectors with random values
    Random rndGen(0);
    forAll(testRho, cellI)
    {
        testRho[cellI] = 0.5 + rndGen.sample01<scalar>();
        testU[cellI] = vector(
            rndGen.sample01<scalar>(),
            rndGen.sample01<scalar>(),
            rndGen.sample01<scalar>()
        );
        testE[cellI] = 0.5 + rndGen.sample01<scalar>();
    }
    
    // Apply the Jacobian matrix to test vectors (A*v)
    // This is a simplified version of the forward operator
    volScalarField Av_rho = fvc::div(testRho*U) + fvc::div(rho*testU);
    volVectorField Av_U("Av_U", fvc::div(testRho*U*U) + fvc::div(rho*testU*U) + fvc::div(rho*U*testU) + fvc::grad(testRho*p/rho));
    volScalarField Av_e("Av_e", fvc::div(testRho*U*thermo.he()) + fvc::div(rho*testU*thermo.he()) + fvc::div(rho*U*testE));
    
    // Apply the adjoint operator to psi (A^T*psi)
    // This is a simplified version of the adjoint operator
    volScalarField ATpsi_rho("ATpsi_rho", fvc::div(psi_U*U) - psi_e*fvc::div(U));
    volVectorField ATpsi_U = fvc::div(psi_rho*U) + fvc::div(psi_U*rho) + fvc::grad(psi_e*p/rho);
    volScalarField ATpsi_e = fvc::div(psi_rho*U) + fvc::div(psi_U*rho*U);
    
    // Calculate the dot products (A^T*psi, v) and (psi, A*v)
    scalar product1 = gSum(
        (ATpsi_rho*testRho).field() +
        (ATpsi_U & testU).field() +
        (ATpsi_e*testE).field()
    )*mesh.V()[0];
    
    scalar product2 = gSum(
        (psi_rho*Av_rho).field() +
        (psi_U & Av_U).field() +
        (psi_e*Av_e).field()
    )*mesh.V()[0];
    
    // Also calculate (dJ/dU, v) directly
    scalar product3 = gSum(
        (dJdrho*testRho).field() +
        (dJdU & testU).field() +
        (dJde*testE).field()
    )*mesh.V()[0];
    
    Info<< "Identity test results:" << nl
        << "  (A^T*psi, v) = " << product1 << nl
        << "  (psi, A*v)   = " << product2 << nl
        << "  (dJ/dU, v)   = " << product3 << nl
        << "  Relative difference 1-2: " << mag(product1 - product2)/max(mag(product1), SMALL) << nl
        << "  Relative difference 1-3: " << mag(product1 - product3)/max(mag(product1), SMALL) << nl
        << "  Relative difference 2-3: " << mag(product2 - product3)/max(mag(product2), SMALL) << nl
        << endl;
    
    // Write the results to a file
    OFstream identityFile(runTime.path()/"verification"/"identityTest.txt");
    identityFile << "Identity Test Results:" << nl
                << "  (A^T*psi, v) = " << product1 << nl
                << "  (psi, A*v)   = " << product2 << nl
                << "  (dJ/dU, v)   = " << product3 << nl
                << "  Relative difference 1-2: " << mag(product1 - product2)/max(mag(product1), SMALL) << nl
                << "  Relative difference 1-3: " << mag(product1 - product3)/max(mag(product1), SMALL) << nl
                << "  Relative difference 2-3: " << mag(product2 - product3)/max(mag(product2), SMALL) << nl;
}

// ===== Finite Difference Verification =====
// Verify the sensitivities by comparing with finite differences
bool performFDVerification = adjointDict.getOrDefault<bool>("performFDVerification", false);

if (performFDVerification)
{
    Info<< "Performing finite difference verification..." << nl << endl;
    
    // Get the parameters for finite difference verification
    scalar fdStepSize = adjointDict.getOrDefault<scalar>("fdStepSize", 1e-6);
    label maxTestPoints = adjointDict.getOrDefault<label>("fdMaxTestPoints", 10);
    word testPatchName = adjointDict.getOrDefault<word>("fdTestPatch", "wall");
    
    // Find the test patch
    label testPatchID = mesh.boundaryMesh().findPatchID(testPatchName);
    
    if (testPatchID == -1)
    {
        Info<< "Warning: Test patch " << testPatchName << " not found, skipping FD verification" << nl << endl;
    }
    else
    {
        const fvPatch& testPatch = mesh.boundary()[testPatchID];
        label nFaces = testPatch.size();
        
        // Limit the number of test points
        label nTestPoints = min(maxTestPoints, nFaces);
        
        Info<< "Using patch " << testPatchName << " with " << nFaces << " faces" << nl
            << "Testing " << nTestPoints << " points with step size " << fdStepSize << nl << endl;
        
        // Create arrays to store the results
        vectorField adjointSens(nTestPoints, vector::zero);
        vectorField fdSens(nTestPoints, vector::zero);
        vectorField facePoints(nTestPoints, vector::zero);
        
        // Get the face centers for the test points
        const vectorField& faceCenters = testPatch.Cf();
        
        // Get the face normals
        const vectorField& faceNormals = testPatch.nf();
        
        // Initialize random number generator for test point selection
        Random rndGen(0);
        
        // Create array of test point indices
        labelList testIndices(nTestPoints);
        
        if (nTestPoints == nFaces)
        {
            // Use all faces
            forAll(testIndices, i)
            {
                testIndices[i] = i;
            }
        }
        else
        {
            // Randomly select faces
            labelHashSet usedIndices;
            
            for (label i = 0; i < nTestPoints; i++)
            {
                label testIdx;
                do
                {
                    testIdx = rndGen.sampleAB<label>(0, nFaces-1);
                } while (usedIndices.found(testIdx));
                
                usedIndices.insert(testIdx);
                testIndices[i] = testIdx;
            }
        }
        
        // Get the base objective function value
        scalar baseObjective = objectiveValue;
        
        // Get the mesh points for the patch
        const labelList& meshPoints = testPatch.meshPoints();
        
        // Create a file to store the results
        OFstream fdFile(runTime.path()/"verification"/"fdVerification.csv");
        fdFile << "# Point,X,Y,Z,Adjoint_dX,Adjoint_dY,Adjoint_dZ,FD_dX,FD_dY,FD_dZ,Rel_Error_X,Rel_Error_Y,Rel_Error_Z" << nl;
        
        // Loop over test points
        forAll(testIndices, i)
        {
            label faceI = testIndices[i];
            
            // Store the face center
            facePoints[i] = faceCenters[faceI];
            
            // Get the adjoint sensitivity at this point
            adjointSens[i] = boundarySensitivity.boundaryField()[testPatchID][faceI];
            
            // Get the face-point mapping
            const labelList& facePts = testPatch.faceFaces()[faceI];
            
            // Loop over directions
            for (direction dir = 0; dir < 3; dir++)
            {
                // Store the original points
                pointField origPoints = mesh.points();
                
                // Create a perturbed mesh by moving points in this direction
                for (const label& ptI : facePts)
                {
                    label meshPtI = meshPoints[ptI];
                    origPoints[meshPtI][dir] += fdStepSize;
                }
                
                // Update the mesh with perturbed points
                mesh.movePoints(origPoints);
                
                // Recompute the objective function with the perturbed mesh
                scalar perturbedObjective = calculateObjective();
                
                // Calculate the finite difference sensitivity
                fdSens[i][dir] = (perturbedObjective - baseObjective) / fdStepSize;
                
                // Reset the mesh to original points
                mesh.movePoints(mesh.points());
            }
            
            // Calculate relative errors
            vector relError;
            for (direction dir = 0; dir < 3; dir++)
            {
                if (mag(adjointSens[i][dir]) > SMALL && mag(fdSens[i][dir]) > SMALL)
                {
                    relError[dir] = mag(adjointSens[i][dir] - fdSens[i][dir]) / 
                                    max(mag(adjointSens[i][dir]), mag(fdSens[i][dir]));
                }
                else
                {
                    relError[dir] = mag(adjointSens[i][dir] - fdSens[i][dir]);
                }
            }
            
            // Write the results to the file
            fdFile << i << ","
                   << facePoints[i].x() << "," << facePoints[i].y() << "," << facePoints[i].z() << ","
                   << adjointSens[i].x() << "," << adjointSens[i].y() << "," << adjointSens[i].z() << ","
                   << fdSens[i].x() << "," << fdSens[i].y() << "," << fdSens[i].z() << ","
                   << relError.x() << "," << relError.y() << "," << relError.z() << nl;
            
            Info<< "Test point " << i << " at " << facePoints[i] << ":" << nl
                << "  Adjoint: " << adjointSens[i] << nl
                << "  FD:      " << fdSens[i] << nl
                << "  Rel Err: " << relError << nl << endl;
        }
        
        // Calculate overall statistics
        scalar avgError = 0.0;
        scalar maxError = 0.0;
        
        forAll(testIndices, i)
        {
            vector relError;
            for (direction dir = 0; dir < 3; dir++)
            {
                if (mag(adjointSens[i][dir]) > SMALL && mag(fdSens[i][dir]) > SMALL)
                {
                    relError[dir] = mag(adjointSens[i][dir] - fdSens[i][dir]) / 
                                    max(mag(adjointSens[i][dir]), mag(fdSens[i][dir]));
                }
                else
                {
                    relError[dir] = mag(adjointSens[i][dir] - fdSens[i][dir]);
                }
                
                avgError += relError[dir];
                maxError = max(maxError, relError[dir]);
            }
        }
        
        avgError /= (3.0 * nTestPoints);
        
        Info<< "Finite difference verification summary:" << nl
            << "  Average relative error: " << avgError << nl
            << "  Maximum relative error: " << maxError << nl << endl;
        
        // Write the summary to a file
        OFstream fdSummary(runTime.path()/"verification"/"fdSummary.txt");
        fdSummary << "Finite Difference Verification Summary:" << nl
                  << "  Number of test points: " << nTestPoints << nl
                  << "  FD step size: " << fdStepSize << nl
                  << "  Average relative error: " << avgError << nl
                  << "  Maximum relative error: " << maxError << nl;
    }
}
