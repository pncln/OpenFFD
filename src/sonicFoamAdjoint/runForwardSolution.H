// Run the forward solution if necessary

// Check if we need to run the forward solution
bool runForward = true;

// Check if a converged solution already exists
if (mag(gSum(phi)) > SMALL && max(mag(U)).value() > SMALL)
{
    Info<< "Forward solution fields present, skipping forward solve" << nl << endl;
    runForward = false;
}
else
{
    Info<< "No converged forward solution found, running sonicFoam" << nl << endl;
}

if (runForward)
{
    Info<< "Starting forward solution..." << nl << endl;
    
    // Time stepping parameters
    scalar forwardTol = 1e-5;
    label maxForwardIter = 1000;
    label forwardIter = 0;
    scalar maxResidual = GREAT;
    
    // PIMPLE control loop
    while (pimple.loop())
    {
        forwardIter++;
        
        Info<< "Forward iteration: " << forwardIter << nl << endl;
        
        // --- Continuity equation
        {
            fvScalarMatrix rhoEqn
            (
                fvm::ddt(rho)
              + fvc::div(phi)
            );

            rhoEqn.solve();
        }
        
        // --- Momentum and energy predictor step
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
            fvm::ddt(rho, thermo.he())
            + fvm::div(phi, thermo.he())
            + fvc::div(phi, 0.5*magSqr(U))
            - fvm::laplacian(turbulence->alphaEff(), thermo.he())
            + (
                U & (
                    fvc::grad(p)
                  + (turbulence->divDevRhoReff(U) & U)
                )
              )
        );
        
        // Add options to energy equation
        fvOptions.constrain(eEqn);
        
        eEqn.relax();
        eEqn.solve();
        
        fvOptions.correct(thermo.he());
        
        // --- Pressure corrector loop
        while (pimple.correct())
        {
            // --- Pressure equation
            volScalarField rAU(1.0/UEqn.A());
            surfaceScalarField rhorAUf("rhorAUf", fvc::interpolate(rho*rAU));
            
            volVectorField HbyA("HbyA", U);
            HbyA = rAU*UEqn.H();
            
            surfaceScalarField phiHbyA
            (
                "phiHbyA",
                fvc::interpolate(rho)*fvc::flux(HbyA)
              + rhorAUf*fvc::ddtCorr(rho, U, phi)
            );
            
            // Update the pressure BCs
            constrainPressure(p, rho, U, phiHbyA, rhorAUf, pimple.dict());
            
            // Non-orthogonal pressure corrector loop
            while (pimple.correctNonOrthogonal())
            {
                // Pressure equation
                fvScalarMatrix pEqn
                (
                    fvm::ddt(psi, p)
                  + fvc::div(phiHbyA)
                  - fvm::laplacian(rhorAUf, p)
                );
                
                pEqn.solve();
                
                if (pimple.finalNonOrthogonalIter())
                {
                    phi = phiHbyA - pEqn.flux();
                }
            }
            
            // Explicitly relax pressure for momentum corrector
            p.relax();
            
            // Momentum corrector
            U = HbyA - rAU*fvc::grad(p);
            U.correctBoundaryConditions();
        }
        
        if (pimple.turbCorr())
        {
            turbulence->correct();
        }
        
        // Calculate some simple convergence metric
        volScalarField contErr("contErr", fvc::div(phi));
        scalar sumLocalContErr = sumMag(contErr.primitiveField())*mesh.time().deltaT().value();
        scalar globalContErr = fvc::domainIntegrate(mag(contErr)).value()*mesh.time().deltaT().value();
        
        maxResidual = globalContErr;
        
        Info<< "time step continuity errors : sum local = " << sumLocalContErr
            << ", global = " << globalContErr << nl << endl;
        
        // Check for convergence
        if (maxResidual < forwardTol)
        {
            Info<< "Forward solution converged!" << nl << endl;
            break;
        }
        
        // Check for maximum iterations
        if (forwardIter >= maxForwardIter)
        {
            Info<< "Forward solution reached maximum iterations!" << nl << endl;
            break;
        }
    }
    
    // Write the solution
    runTime++;
    Info<< "Writing forward solution at time " << runTime.timeName() << nl << endl;
    
    // Write the fields
    thermo.p().write();
    thermo.T().write();
    thermo.he().write();
    rho.write();
    U.write();
    phi.write();
}

// Update the cached fields needed for the adjoint solution
c = sqrt(thermo.Cp()/thermo.Cv() * p/rho);
gradU = fvc::grad(U);
mu = thermo.mu();
rhoU = rho*U;

    // Calculate objective function - will be implemented in setupObjectiveFunction.H
    Info<< "Objective function calculation will be performed during adjoint solve" << nl << endl;
