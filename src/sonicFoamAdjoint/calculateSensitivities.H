// Calculate sensitivities with respect to mesh coordinates

Info<< "Calculating mesh sensitivities for objective: " << objectiveName << nl << endl;

// Initialize the sensitivity field
meshSensitivity = dimensionedVector("zero", meshSensitivity.dimensions(), vector::zero);

// We need to calculate: dJ/dx = dJ/dU * dU/dx + dJ/dp * dp/dx + ...

// Create sensitivity contributions from each state variable
volVectorField gradRho = fvc::grad(rho);
volTensorField gradP = fvc::grad(p);
volTensorField gradT = fvc::grad(T);

// Calculate the geometric sensitivity contributions
// This is a simplified version; the full version would include all terms from the PDE

// First, let's handle the boundaries, which is where most of the sensitivities come from
// for aerodynamic shape optimization

forAll(mesh.boundaryMesh(), patchI)
{
    const fvPatch& patch = mesh.boundary()[patchI];
    
    // Only compute sensitivities on wall patches
    if (U.boundaryField()[patchI].type() == "noSlip" || 
        U.boundaryField()[patchI].type() == "fixedValue" ||
        patch.name().find("wall") != std::string::npos ||
        patch.name().find("Wing") != std::string::npos ||
        patch.name().find("wing") != std::string::npos ||
        patch.name().find("Airfoil") != std::string::npos ||
        patch.name().find("airfoil") != std::string::npos)
    {
        Info<< "Computing sensitivities on patch: " << patch.name() << nl << endl;
        
        // Get patch geometry
        const vectorField& Sf = patch.Sf();
        const scalarField& magSf = patch.magSf();
        const vectorField& Cf = patch.Cf();
        const vectorField& nf = Sf / magSf;
        
        // Get the adjoint variables on the patch
        const scalarField& psi_rho_p = psi_rho.boundaryField()[patchI];
        const vectorField& psi_U_p = psi_U.boundaryField()[patchI];
        const scalarField& psi_e_p = psi_e.boundaryField()[patchI];
        
        // Get the primal variables on the patch
        const scalarField& p_p = p.boundaryField()[patchI];
        const vectorField& U_p = U.boundaryField()[patchI];
        const scalarField& rho_p = rho.boundaryField()[patchI];
        const scalarField& T_p = T.boundaryField()[patchI];
        
        // Placeholder for patch sensitivities
        vectorField patchSensitivity(patch.size(), vector::zero);
        
        // Calculate the sensitivities on this patch
        if (objectiveName == "drag" || objectiveName == "lift")
        {
            // Get the force direction vector
            vector direction = objDict.get<vector>("direction");
            direction /= mag(direction); // Normalize
            
            // For pressure contribution: dJ/dx = (dJ/dp) * (dp/dx)
            // For drag/lift, dJ/dp = n.d on the surface
            // The derivative dp/dx depends on the surface deformation
            
            // Calculate the pressure contribution to the sensitivity
            forAll(patch, faceI)
            {
                // For pressure contribution: Surface normal perturbation method
                // This is a simplified approach; a more complete implementation would 
                // include shape derivatives of the viscous terms as well
                
                // The sensitivity is primarily due to the change in the surface normal
                // when the surface is perturbed
                patchSensitivity[faceI] = p_p[faceI] * (direction - nf[faceI]*(nf[faceI] & direction));
                
                // Note: This is simplified. For viscous flows, we would need to 
                // include contributions from viscous stresses as well.
            }
        }
        else if (objectiveName == "moment")
        {
            // Get the moment axis direction
            vector direction = objDict.get<vector>("direction");
            direction /= mag(direction); // Normalize
            
            // Get the moment center
            vector center = objDict.get<vector>("center");
            
            forAll(patch, faceI)
            {
                // Lever arm
                vector r = Cf[faceI] - center;
                
                // Calculate sensitivity due to change in surface normal
                // This affects the pressure force direction
                vector normalSens = p_p[faceI] * (direction - nf[faceI]*(nf[faceI] & direction));
                
                // Calculate sensitivity due to change in lever arm
                // This affects the moment arm
                vector leverSens = vector::zero;
                
                // Combine sensitivities
                patchSensitivity[faceI] = normalSens + leverSens;
            }
        }
        // Add similar implementation for other objective functions
        
        // Store the sensitivities on this patch
        forAll(patch, faceI)
        {
            // Convert face sensitivity to cell sensitivity
            label cellI = patch.faceCells()[faceI];
            meshSensitivity[cellI] = patchSensitivity[faceI];
            
            // Also store in the boundary sensitivity field for export
            boundarySensitivity.boundaryFieldRef()[patchI][faceI] = patchSensitivity[faceI];
        }
    }
}

// Optional: Smoothing of sensitivities to avoid noisy surfaces
// Using OpenFOAM's fvc::smooth utility
fvc::smooth(meshSensitivity, 0.5);

// Calculate the L2 norm of the sensitivity field
scalar sensNorm = Foam::sqrt(gSum(magSqr(meshSensitivity)) / meshSensitivity.size());

Info<< "Sensitivity L2 norm: " << sensNorm << nl << endl;

// Write the sensitivity fields
meshSensitivity.write();
boundarySensitivity.write();

// Print maximum sensitivity components
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

Info<< "Maximum sensitivity: " << maxSens << nl
    << "Minimum sensitivity: " << minSens << nl << endl;
