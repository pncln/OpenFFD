// Run the forward solution if necessary

// Check if we need to run the forward solution
bool runForward = true;

// Check if a converged solution already exists
if (mag(phi) > SMALL && max(mag(U)).value() > SMALL)
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
        
        #include "rhoEqn.H"
        
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
            ==
            fvOptions(rho, thermo.he())
        );
        
        eEqn.relax();
        eEqn.solve();
        
        fvOptions.correct(thermo.he());
        
        // --- Pressure corrector loop
        while (pimple.correct())
        {
            #include "pEqn.H"
        }
        
        if (pimple.turbCorr())
        {
            turbulence->correct();
        }
        
        // Calculate residuals
        volScalarField rhoRes = fvc::ddt(rho) + fvc::div(phi);
        volVectorField URes = fvc::ddt(rho, U) + fvc::div(phi, U) + fvc::grad(p) + turbulence->divDevRhoReff(U);
        volScalarField eRes = fvc::ddt(rho, thermo.he()) + fvc::div(phi, thermo.he()) - fvc::laplacian(turbulence->alphaEff(), thermo.he());
        
        // Get maximum residuals
        scalar maxRhoRes = gMax(mag(rhoRes));
        scalar maxURes = gMax(mag(URes));
        scalar maxERes = gMax(mag(eRes));
        
        maxResidual = max(maxRhoRes, max(maxURes, maxERes));
        
        Info<< "Maximum residual: " << maxResidual << nl << endl;
        
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

// Calculate the objective function
objectiveValue = calculateObjective();
Info<< "Objective function value: " << objectiveValue << nl << endl;
