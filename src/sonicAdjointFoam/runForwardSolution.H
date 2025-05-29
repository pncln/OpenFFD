// Run the forward solution to get the baseline flow field
Info<< "Running forward solution...\n" << endl;

// Use the PIMPLE control object created in createControl.H

// Initialize flow statistics
scalar initialMass = fvc::domainIntegrate(rho).value();
scalar sumLocalContErr = 0.0;
scalar globalContErr = 0.0;

// Main time loop for forward solution
while (runTime.run())
{
    #include "createTimeControls.H"
    #include "compressibleCourantNo.H"
    #include "setDeltaT.H"
    
    ++runTime;
    
    Info<< "Forward time = " << runTime.timeName() << nl << endl;
    
    // Store previous iteration values for convergence checking
    p.storePrevIter();
    rho.storePrevIter();
    
    // --- Pressure-velocity PIMPLE corrector loop
    while (pimple.loop())
    {
        // --- Solve the approximate momentum equation
        fvVectorMatrix UEqn
        (
            fvm::ddt(rho, U)
          + fvm::div(phi, U)
          + turbulence->divDevRhoReff(U)
        );
        
        UEqn.relax();
        
        if (pimple.momentumPredictor())
        {
            solve(UEqn == -fvc::grad(p));
        }
        
        // --- Energy equation
        fvScalarMatrix eEqn
        (
            fvm::ddt(rho, e)
          + fvm::div(phi, e)
          + fvc::div(phi, 0.5*magSqr(U))
          - fvm::laplacian(alphaEff, e)
          - fvc::div(U & turbulence->devRhoReff())
        );
        
        eEqn.relax();
        eEqn.solve();
        
        // --- Pressure corrector loop
        while (pimple.correct())
        {
            // Thermodynamic updates
            thermo.correct();
            rho = thermo.rho();
            
            // Update pressure fluxes
            tmp<surfaceScalarField> trhoUf = fvc::interpolate(rho*U);
            surfaceScalarField& rhoUf = trhoUf.ref();
            
            tmp<surfaceVectorField> trhoUfU = rhoUf*fvc::interpolate(U);
            surfaceVectorField& rhoUfU = trhoUfU.ref();
            
            // Solve the continuity equation
            fvScalarMatrix pEqn
            (
                fvm::ddt(psi, p)
              + fvc::div(rhoUf)
              - fvm::laplacian(rho/psi, p)
            );
            
            pEqn.solve();
            
            // Update velocity
            U = UEqn.H()/UEqn.A() - fvc::grad(p)/UEqn.A();
            U.correctBoundaryConditions();
            
            // Update fluxes
            phi = fvc::interpolate(rho)*(fvc::interpolate(U) & mesh.Sf());
            
            // Calculate continuity errors
            #include "compressibleContinuityErrs.H"
        }
        
        // Update Mach number
        Mach = mag(U)/sqrt(thermo.gamma()*thermo.Cp()*T/thermo.Cv());
        
        // Update turbulence model
        turbulence->correct();
    }
    
    // Calculate the objective function value
    #include "calculateObjective.H"
    
    // Check for convergence
    scalar residual = max(
        max(p.relativeError(), rho.relativeError()),
        mag(objectiveValue - objectiveValueOld)/max(SMALL, mag(objectiveValueOld))
    );
    
    Info<< "Continuity error: " << globalContErr << nl
        << "Max residual: " << residual << nl
        << "Objective function value: " << objectiveValue << nl << endl;
    
    if (residual < 1e-6 && runTime.timeIndex() > 1)
    {
        Info<< "Solution converged" << endl;
        break;
    }
    
    runTime.write();
}

// Final write
if (runTime.timeIndex() > 0)
{
    runTime.write();
}

Info<< "Forward solution completed" << nl
    << "Final objective function value: " << objectiveValue << nl << endl;
