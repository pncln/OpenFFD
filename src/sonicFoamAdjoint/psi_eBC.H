/*---------------------------------------------------------------------------*\
  Boundary conditions for adjoint energy (psi_e)

  Implements the mathematically consistent adjoint boundary conditions 
  derived from the continuous adjoint formulation.
\*---------------------------------------------------------------------------*/

// Get references to boundary fields
volScalarField::Boundary& psi_eBf = psi_e.boundaryFieldRef();

// Loop over all patches
forAll(mesh.boundaryMesh(), patchi)
{
    const fvPatch& patch = mesh.boundary()[patchi];
    const word& patchType = patch.type();
    const word& patchName = patch.name();
    
    // Get boundary condition types from the primal solution
    const word UBCType = U.boundaryField()[patchi].type();
    const word TBCType = T.boundaryField()[patchi].type();
    
    // Wall boundary conditions
    if (UBCType == "noSlip" || UBCType == "fixedValue" || 
        patchName.find("wall") != string::npos ||
        patchType == "wall")
    {
        // For isothermal walls: fixed value condition for psi_e
        if (TBCType == "fixedValue")
        {
            if (psi_eBf[patchi].type() != "fixedValue")
            {
                Info<< "Setting psi_e BC to fixedValue for isothermal wall patch " << patchName << endl;
                
                // Convert to fixedValue if not already
                psi_eBf[patchi] = fvPatchScalarField::New("fixedValue", patch, psi_e);
                psi_eBf[patchi] == 0.0;
            }
        }
        // For adiabatic walls: zero gradient condition for psi_e
        else if (TBCType == "zeroGradient")
        {
            if (psi_eBf[patchi].type() != "zeroGradient")
            {
                Info<< "Setting psi_e BC to zeroGradient for adiabatic wall patch " << patchName << endl;
                
                // Convert to zeroGradient if not already
                psi_eBf[patchi] = fvPatchScalarField::New("zeroGradient", patch, psi_e);
            }
        }
        // Default to fixedValue for walls if unknown temperature BC
        else
        {
            if (psi_eBf[patchi].type() != "fixedValue")
            {
                Info<< "Setting psi_e BC to fixedValue (default) for wall patch " << patchName << endl;
                
                // Convert to fixedValue if not already
                psi_eBf[patchi] = fvPatchScalarField::New("fixedValue", patch, psi_e);
                psi_eBf[patchi] == 0.0;
            }
        }
    }
    
    // Inlet boundary conditions
    else if (patchName.find("inlet") != string::npos || 
             patchName.find("Inlet") != string::npos ||
             patchType == "patch" && UBCType == "fixedValue")
    {
        // For inlets with fixed temperature: fixedValue BC for psi_e
        if (TBCType == "fixedValue")
        {
            if (psi_eBf[patchi].type() != "fixedValue")
            {
                Info<< "Setting psi_e BC to fixedValue for inlet patch " << patchName << endl;
                
                // Convert to fixedValue if not already
                psi_eBf[patchi] = fvPatchScalarField::New("fixedValue", patch, psi_e);
                psi_eBf[patchi] == 0.0;
            }
        }
        // For inlets with other temperature BCs: use zeroGradient
        else
        {
            if (psi_eBf[patchi].type() != "zeroGradient")
            {
                Info<< "Setting psi_e BC to zeroGradient for inlet patch " << patchName << endl;
                
                // Convert to zeroGradient if not already
                psi_eBf[patchi] = fvPatchScalarField::New("zeroGradient", patch, psi_e);
            }
        }
    }
    
    // Outlet boundary conditions
    else if (patchName.find("outlet") != string::npos || 
             patchName.find("Outlet") != string::npos)
    {
        // For outlets: use zeroGradient for psi_e
        if (psi_eBf[patchi].type() != "zeroGradient")
        {
            Info<< "Setting psi_e BC to zeroGradient for outlet patch " << patchName << endl;
            
            // Convert to zeroGradient if not already
            psi_eBf[patchi] = fvPatchScalarField::New("zeroGradient", patch, psi_e);
        }
        
        // If this patch is part of an objective like temperature uniformity, add contribution
        if (objectiveName == "uniformity" && objDict.found("patches"))
        {
            const wordList& objPatches = objDict.get<wordList>("patches");
            const word& fieldName = objDict.get<word>("field");
            
            if (findIndex(objPatches, patchName) != -1 && fieldName == "T")
            {
                // Calculate the average temperature
                const scalarField& Tb = T.boundaryField()[patchi];
                const scalarField& magSf = patch.magSf();
                scalar Tavg = gSum(Tb * magSf) / gSum(magSf);
                scalar scale = objDict.getOrDefault<scalar>("scale", 1.0);
                
                // For temperature uniformity objective: adjoint BC depends on deviation from average
                if (psi_eBf[patchi].type() != "fixedGradient")
                {
                    psi_eBf[patchi] = fvPatchScalarField::New("fixedGradient", patch, psi_e);
                }
                
                // Apply the gradient based on the uniformity objective
                fixedGradientFvPatchScalarField& gradPsi_e = 
                    refCast<fixedGradientFvPatchScalarField>(psi_eBf[patchi]);
                
                // Gradient is proportional to temperature deviation from average
                gradPsi_e.gradient() = 2.0 * scale * (Tb - Tavg);
            }
        }
    }
    
    // Symmetry planes
    else if (patchType == "symmetry" || patchType == "symmetryPlane" || 
             patchName.find("symm") != string::npos)
    {
        // Symmetry condition: use OpenFOAM's symmetry BC
        if (psi_eBf[patchi].type() != "symmetry")
        {
            Info<< "Setting psi_e BC to symmetry for patch " << patchName << endl;
            
            // Convert to symmetry BC if not already
            psi_eBf[patchi] = fvPatchScalarField::New("symmetry", patch, psi_e);
        }
    }
    
    // Far-field boundaries
    else if (patchName.find("farField") != string::npos || 
             patchName.find("far") != string::npos)
    {
        // For far-field boundaries: characteristic-based BCs
        // Simplified implementation: use zeroGradient
        if (psi_eBf[patchi].type() != "zeroGradient")
        {
            Info<< "Setting psi_e BC to zeroGradient for patch " << patchName << endl;
            
            // Convert to zeroGradient if not already
            psi_eBf[patchi] = fvPatchScalarField::New("zeroGradient", patch, psi_e);
        }
    }
    
    // Default: use zeroGradient for any unhandled cases
    else
    {
        if (psi_eBf[patchi].type() != "zeroGradient")
        {
            Info<< "Setting psi_e BC to zeroGradient (default) for patch " << patchName << endl;
            
            // Convert to zeroGradient if not already
            psi_eBf[patchi] = fvPatchScalarField::New("zeroGradient", patch, psi_e);
        }
    }
}
