// Create adjoint variables for the adjoint solver

Info<< "Creating adjoint fields\n" << endl;

// Adjoint variable for mass conservation (continuity) equation
volScalarField psi_rho
(
    IOobject
    (
        "psi_rho",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("psi_rho", dimMass/dimVolume, 0.0)
);

// Adjoint variable for momentum conservation equation
volVectorField psi_U
(
    IOobject
    (
        "psi_U",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedVector("psi_U", dimMass/(dimLength*dimLength), vector(0, 0, 0))
);

// Adjoint variable for energy conservation equation
volScalarField psi_e
(
    IOobject
    (
        "psi_e",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("psi_e", dimMass/(dimLength*dimTime*dimTime), 0.0)
);

// Flux of adjoint variables
surfaceScalarField psi_phi
(
    IOobject
    (
        "psi_phi",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    fvc::flux(psi_U)
);

// Sensitivity field for mesh movement
volVectorField meshSensitivity
(
    IOobject
    (
        "meshSensitivity",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedVector("meshSensitivity", dimPressure, vector(0, 0, 0))
);

// Boundary sensitivities for specific wall patches
// This will store the sensitivity on boundary walls
volVectorField boundarySensitivity
(
    IOobject
    (
        "boundarySensitivity",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedVector("boundarySensitivity", dimPressure, vector(0, 0, 0))
);

// Create relaxation factors for adjoint equations
scalar psi_rhoRelaxFactor = 0.3;
scalar psi_URelaxFactor = 0.7;
scalar psi_eRelaxFactor = 0.3;

// Store previous iteration values for under-relaxation
volScalarField psi_rho_0
(
    IOobject
    (
        "psi_rho_0",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    psi_rho
);

volVectorField psi_U_0
(
    IOobject
    (
        "psi_U_0",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    psi_U
);

volScalarField psi_e_0
(
    IOobject
    (
        "psi_e_0",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    psi_e
);

// Adjoint residuals for convergence monitoring
volScalarField psi_rhoRes
(
    IOobject
    (
        "psi_rhoRes",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("psi_rhoRes", dimMass/(dimVolume*dimTime), 0.0)
);

volVectorField psi_URes
(
    IOobject
    (
        "psi_URes",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedVector("psi_URes", dimMass/(dimLength*dimLength*dimTime), vector(0, 0, 0))
);

volScalarField psi_eRes
(
    IOobject
    (
        "psi_eRes",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("psi_eRes", dimMass/(dimLength*dimTime*dimTime*dimTime), 0.0)
);

// Maximum residual tracking for convergence monitoring
scalar maxPsi_rhoRes = GREAT;
scalar maxPsi_URes = GREAT;
scalar maxPsi_eRes = GREAT;

// Iteration counter for the adjoint solution
label adjointIter = 0;

// Maximum number of iterations for the adjoint solution
label maxAdjointIter = 1000;

// Convergence tolerance for the adjoint solution
scalar adjointTol = 1e-6;
