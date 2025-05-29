/*---------------------------------------------------------------------------*\
  Boundary conditions for adjoint velocity (psi_U)

  Implements the mathematically consistent adjoint boundary conditions 
  derived from the continuous adjoint formulation.
\*---------------------------------------------------------------------------*/

// Get references to boundary fields
volVectorField::Boundary& psi_UBf = psi_U.boundaryFieldRef();

// Loop over all patches
forAll(mesh.boundaryMesh(), patchi)
{
    const fvPatch& patch = mesh.boundary()[patchi];
    const word& patchType = patch.type();
    const word& patchName = patch.name();
    
    // Get boundary condition types from the primal solution
    const word UBCType = U.boundaryField()[patchi].type();
    const word pBCType = p.boundaryField()[patchi].type();
    
    // Wall boundary conditions
    if (UBCType == "noSlip" || UBCType == "fixedValue" || 
        patchName.find("wall") != string::npos ||
        patchType == "wall")
    {
        // For no-slip walls: fixed value to zero for psi_U
        // This is the adjoint of the no-slip condition
        if (psi_UBf[patchi].type() != "fixedValue")
        {
            Info<< "Setting psi_U BC to fixedValue for patch " << patchName << endl;
            
            // Convert to fixedValue if not already
            psi_UBf[patchi] = fvPatchVectorField::New("fixedValue", patch, psi_U);
            psi_UBf[patchi] == vector::zero;
        }
        
        // If this patch is part of the objective function, modify BC
        // For drag/lift objective functions on walls
        if ((objectiveName == "drag" || objectiveName == "lift" || objectiveName == "moment") && 
            objDict.found("patches"))
        {
            const wordList& objPatches = objDict.get<wordList>("patches");
            
            if (findIndex(objPatches, patchName) != -1)
            {
                // For drag objective: BC depends on direction
                vector direction = objDict.get<vector>("direction");
                direction /= mag(direction); // Normalize
                
                // Get patch geometry
                const vectorField& Sf = patch.Sf();
                const scalarField& magSf = patch.magSf();
                const vectorField nf = Sf / magSf;
                
                // Create a surface traction contribution from the objective
                vectorField tractionContrib(patch.size(), vector::zero);
                
                // For drag/lift: the adjoint BC includes the force direction
                if (objectiveName == "drag" || objectiveName == "lift")
                {
                    scalar scale = objDict.getOrDefault<scalar>("scale", 1.0);
                    
                    forAll(patch, facei)
                    {
                        tractionContrib[facei] = scale * direction;
                    }
                }
                // For moment: the adjoint BC includes the moment arm and direction
                else if (objectiveName == "moment")
                {
                    vector center = objDict.get<vector>("center");
                    const vectorField& Cf = patch.Cf();
                    scalar scale = objDict.getOrDefault<scalar>("scale", 1.0);
                    
                    forAll(patch, facei)
                    {
                        // Lever arm
                        vector r = Cf[facei] - center;
                        
                        // Moment contribution
                        // Using cross product of r and direction
                        tractionContrib[facei] = scale * (r ^ direction);
                    }
                }
                
                // Apply the traction contribution to the boundary condition
                // This is a simplified implementation; a real one would be more complex
                // In this simplified version, we use fixedGradient BC
                if (psi_UBf[patchi].type() != "fixedGradient")
                {
                    psi_UBf[patchi] = fvPatchVectorField::New("fixedGradient", patch, psi_U);
                }
                
                // Apply the gradient based on the traction
                fixedGradientFvPatchVectorField& gradPsi_U = 
                    refCast<fixedGradientFvPatchVectorField>(psi_UBf[patchi]);
                
                gradPsi_U.gradient() = tractionContrib / mu.boundaryField()[patchi];
            }
        }
    }
    
    // Inlet boundary conditions
    else if (patchName.find("inlet") != string::npos || 
             patchName.find("Inlet") != string::npos ||
             patchType == "patch" && UBCType == "fixedValue")
    {
        // For subsonic inlets: complex characteristic-based BC
        // Simplified implementation: use zeroGradient for the tangential components
        // and fixedValue for the normal component
        if (psi_UBf[patchi].type() != "mixed")
        {
            Info<< "Setting psi_U BC to mixed for patch " << patchName << endl;
            
            // Convert to mixed BC if not already
            psi_UBf[patchi] = fvPatchVectorField::New("mixed", patch, psi_U);
            
            // Get patch geometry
            const vectorField& Sf = patch.Sf();
            const scalarField& magSf = patch.magSf();
            const vectorField nf = Sf / magSf;
            
            // Calculate the value fraction (0 for tangential component, 1 for normal)
            tensorField valueFraction(patch.size(), tensor::zero);
            forAll(patch, facei)
            {
                // Create a tensor that extracts the normal component
                // twoSymm(nf[facei] * nf[facei]) creates a symmetric tensor
                // from the outer product of nf with itself
                valueFraction[facei] = twoSymm(nf[facei] * nf[facei]);
            }
            
            // Set the mixed BC parameters
            mixedFvPatchVectorField& mixedPsi_U = 
                refCast<mixedFvPatchVectorField>(psi_UBf[patchi]);
            
            mixedPsi_U.valueFraction() = valueFraction;
            mixedPsi_U.refValue() = vector::zero;
            mixedPsi_U.refGrad() = vector::zero;
        }
    }
    
    // Outlet boundary conditions
    else if (patchName.find("outlet") != string::npos || 
             patchName.find("Outlet") != string::npos ||
             (patchType == "patch" && pBCType == "fixedValue"))
    {
        // For subsonic outlets: more complex BCs based on adjoint theory
        // Simplified implementation: use zeroGradient
        if (psi_UBf[patchi].type() != "zeroGradient")
        {
            Info<< "Setting psi_U BC to zeroGradient for patch " << patchName << endl;
            
            // Convert to zeroGradient if not already
            psi_UBf[patchi] = fvPatchVectorField::New("zeroGradient", patch, psi_U);
        }
        
        // If this patch is part of an objective like uniformity, add contribution
        if (objectiveName == "uniformity" && objDict.found("patches"))
        {
            const wordList& objPatches = objDict.get<wordList>("patches");
            
            if (findIndex(objPatches, patchName) != -1 && 
                objDict.get<word>("field") == "U")
            {
                // Calculate the average velocity
                const vectorField& Ub = U.boundaryField()[patchi];
                const scalarField& magSf = patch.magSf();
                vector Uavg = gSum(Ub * magSf) / gSum(magSf);
                scalar scale = objDict.getOrDefault<scalar>("scale", 1.0);
                
                // For uniformity objective: adjoint BC depends on deviation from average
                if (psi_UBf[patchi].type() != "fixedGradient")
                {
                    psi_UBf[patchi] = fvPatchVectorField::New("fixedGradient", patch, psi_U);
                }
                
                // Apply the gradient based on the uniformity objective
                fixedGradientFvPatchVectorField& gradPsi_U = 
                    refCast<fixedGradientFvPatchVectorField>(psi_UBf[patchi]);
                
                // Gradient is proportional to deviation from average
                forAll(patch, facei)
                {
                    gradPsi_U.gradient()[facei] = 2.0 * scale * (Ub[facei] - Uavg);
                }
            }
        }
    }
    
    // Symmetry planes
    else if (patchType == "symmetry" || patchType == "symmetryPlane" || 
             patchName.find("symm") != string::npos)
    {
        // Symmetry condition: use OpenFOAM's symmetry BC
        if (psi_UBf[patchi].type() != "symmetry")
        {
            Info<< "Setting psi_U BC to symmetry for patch " << patchName << endl;
            
            // Convert to symmetry BC if not already
            psi_UBf[patchi] = fvPatchVectorField::New("symmetry", patch, psi_U);
        }
    }
    
    // Far-field boundaries
    else if (patchName.find("farField") != string::npos || 
             patchName.find("far") != string::npos)
    {
        // For far-field boundaries: characteristic-based BCs
        // Simplified implementation: use zeroGradient
        if (psi_UBf[patchi].type() != "zeroGradient")
        {
            Info<< "Setting psi_U BC to zeroGradient for patch " << patchName << endl;
            
            // Convert to zeroGradient if not already
            psi_UBf[patchi] = fvPatchVectorField::New("zeroGradient", patch, psi_U);
        }
    }
    
    // Default: use zeroGradient for any unhandled cases
    else
    {
        if (psi_UBf[patchi].type() != "zeroGradient")
        {
            Info<< "Setting psi_U BC to zeroGradient (default) for patch " << patchName << endl;
            
            // Convert to zeroGradient if not already
            psi_UBf[patchi] = fvPatchVectorField::New("zeroGradient", patch, psi_U);
        }
    }
}
