// Store the previous value of the objective function
scalar objectiveValueOld = objectiveValue;
objectiveValue = 0.0;

// Calculate the objective function based on the specified type
// Note: objectiveType is a global variable defined in globalVariables.H
if (objectiveType == word("drag") || objectiveType == word("lift") || objectiveType == word("force"))
{
    vector resultForce(vector::zero);
    
    // Calculate force on specified patches
    forAll(patchNames, patchI)
    {
        const label patchID = mesh.boundaryMesh().findPatchID(patchNames[patchI]);
        
        if (patchID == -1)
        {
            FatalErrorInFunction
                << "Cannot find patch " << patchNames[patchI] << nl
                << exit(FatalError);
        }
        
        // Get patch info
        const fvPatch& patch = mesh.boundary()[patchID];
        const vectorField& Sf = patch.Sf();
        const scalarField& magSf = patch.magSf();
        
        // Calculate pressure and viscous forces
        const scalarField& pPatch = p.boundaryField()[patchID];
        const vectorField& UPatch = U.boundaryField()[patchID];
        const scalarField& rhoPatch = rho.boundaryField()[patchID];
        const symmTensorField& tauPatch = turbulence->devRhoReff()->boundaryField()[patchID];
        
        // Normal vectors
        vectorField nf(Sf/magSf);
        
        // Calculate pressure force
        vector pressureForce = gSum(pPatch*Sf);
        
        // Calculate viscous force
        vector viscousForce = gSum(tauPatch & Sf);
        
        // Total force on this patch
        resultForce += pressureForce + viscousForce;
    }
    
    // Calculate the objective function as the force in the specified direction
    objectiveValue = (resultForce & forceDirection) * objFuncScale;
}
else if (objectiveType == word("pressure_uniformity"))
{
    scalar sumVariance = 0.0;
    scalar sumArea = 0.0;
    
    // Calculate pressure uniformity on specified patches
    forAll(patchNames, patchI)
    {
        const label patchID = mesh.boundaryMesh().findPatchID(patchNames[patchI]);
        
        if (patchID == -1)
        {
            FatalErrorInFunction
                << "Cannot find patch " << patchNames[patchI] << nl
                << exit(FatalError);
        }
        
        // Get patch info
        const fvPatch& patch = mesh.boundary()[patchID];
        const scalarField& magSf = patch.magSf();
        
        // Get pressure on patch
        const scalarField& pPatch = p.boundaryField()[patchID];
        
        // Calculate average pressure on this patch
        scalar avgPressure = gSum(pPatch*magSf)/gSum(magSf);
        
        // Calculate variance from average
        scalarField variance(sqr(pPatch - avgPressure));
        
        // Add to total variance (weighted by area)
        sumVariance += gSum(variance*magSf);
        sumArea += gSum(magSf);
    }
    
    // Normalize by total area
    objectiveValue = (sumVariance/max(SMALL, sumArea)) * objFuncScale;
}
else if (objectiveType == word("entropy_generation"))
{
    scalar totalEntropyGeneration = 0.0;
    
    // Volume integral of entropy generation
    volScalarField gradTSqr
    (
        IOobject
        (
            "gradTSqr",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        magSqr(fvc::grad(T))
    );
    volTensorField gradU
    (
        IOobject
        (
            "gradU",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        fvc::grad(U)
    );
    
    // Define Prandtl numbers if not already available
    const scalar Prandtl = 0.7;  // Laminar Prandtl number for air
    const scalar Prt = 0.9;      // Turbulent Prandtl number
    const volScalarField dissipation = (mu + mut)*magSqr(symm(gradU)) + 
                                      (mu/Prandtl + mut/Prt)*gradTSqr/(sqr(T) + SMALL);
    
    totalEntropyGeneration = fvc::domainIntegrate(dissipation).value();
    
    objectiveValue = totalEntropyGeneration * objFuncScale;
}
else
{
    FatalErrorInFunction
        << "Unknown objective function type: " << objectiveType << nl
        << "Valid options are: drag, lift, force, pressure_uniformity, entropy_generation" << nl
        << exit(FatalError);
}

Info<< "Objective function (" << objectiveType << "): " << objectiveValue << endl;
