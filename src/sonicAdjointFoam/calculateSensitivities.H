// Calculate sensitivities based on adjoint solution
Info<< "Calculating mesh sensitivities..." << endl;

// Reset the sensitivity field
meshSensitivity = dimensionedVector("zero", meshSensitivity.dimensions(), vector::zero);

// We need to calculate: dJ/dx = dJ/dU * dU/dx + dJ/dp * dp/dx + ...

// Create sensitivity contributions from each state variable
volVectorField gradRho("gradRho", fvc::grad(rho));
volVectorField gradP("gradP", fvc::grad(p));
volVectorField gradT("gradT", fvc::grad(T));

// Calculate the geometric sensitivity contributions
// This is a simplified version; the full version would include all terms from the PDE
volVectorField rhoSens = psi_rho * gradRho;
volVectorField pSens = (psi_e/rho) * gradP;
volVectorField TSens = (psi_e*(Cp - Cv)/Cv) * gradT;

// Compute gradients of velocity components
volTensorField gradUx("gradUx", fvc::grad(U.component(0)));
volTensorField gradUy("gradUy", fvc::grad(U.component(1)));
volTensorField gradUz("gradUz", fvc::grad(U.component(2)));

// Velocity sensitivity contributions
volVectorField USensX = psi_U.component(0) * gradUx;
volVectorField USensY = psi_U.component(1) * gradUy;
volVectorField USensZ = psi_U.component(2) * gradUz;

// Combine sensitivities from all state variables
meshSensitivity = rhoSens + pSens + TSens + USensX + USensY + USensZ;

// Normalize the sensitivity field
scalar maxSens = gMax(mag(meshSensitivity));
if (maxSens > SMALL)
{
    Info<< "Normalizing sensitivities by maximum value: " << maxSens << endl;
    meshSensitivity /= maxSens;
}

// Optional sensitivity smoothing
if (smoothSensitivity)
{
    Info<< "Smoothing sensitivities..." << endl;
    
    // Create temporary field for smoothing
    volVectorField smoothedSens = meshSensitivity;
    
    // Laplacian smoothing
    label nSmoothingIters = 5;
    scalar smoothingCoeff = 0.5;
    
    for (label iter = 0; iter < nSmoothingIters; iter++)
    {
        // Store original field
        volVectorField origSens = smoothedSens;
        
        // Apply Laplacian smoothing
        smoothedSens = fvc::laplacian(dimensionedScalar("dt", dimTime, 1.0), origSens);
        smoothedSens = origSens + smoothingCoeff*smoothedSens;
        
        // Preserve boundary values
        forAll(mesh.boundary(), patchI)
        {
            const fvPatch& patch = mesh.boundary()[patchI];
            smoothedSens.boundaryFieldRef()[patchI] = meshSensitivity.boundaryField()[patchI];
        }
    }
    
    // Update sensitivity field with smoothed values
    meshSensitivity = smoothedSens;
}

// Calculate the L2 norm of the sensitivity field
volScalarField sensNormField("sensNormField", magSqr(meshSensitivity));
scalar sensL2Norm = sqrt(gSum(sensNormField.primitiveField()*mesh.V())/gSum(mesh.V()));

Info<< "Sensitivity calculation complete:" << nl
    << "  Max sensitivity magnitude: " << gMax(mag(meshSensitivity)) << nl
    << "  L2 norm of sensitivity: " << sensL2Norm << nl << endl;

// Write the sensitivity field
runTime.write();
