// Create base fields similar to sonicFoam
Info<< "Reading thermophysical properties\n" << endl;

autoPtr<psiThermo> pThermo
(
    psiThermo::New(mesh)
);
psiThermo& thermo = pThermo();
thermo.validate(args.executable(), "h", "e");

volScalarField& p = thermo.p();

// Access T and psi but mark as maybe unused to prevent warnings
volScalarField& T = thermo.T(); (void)T; // Mark as used
const volScalarField& psi = thermo.psi(); (void)psi; // Mark as used

// Define rho from thermo
const volScalarField& rho = thermo.rho();

Info<< "Reading field U\n" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

#include "compressibleCreatePhi.H"

// Set up pressure reference cell/value if needed
label pRefCell = 0;
scalar pRefValue = 0.0;
setRefCell(p, pimple.dict(), pRefCell, pRefValue);

mesh.setFluxRequired(p.name());

Info<< "Creating turbulence model\n" << endl;
autoPtr<compressible::turbulenceModel> turbulence
(
    compressible::turbulenceModel::New
    (
        rho,
        U,
        phi,
        thermo
    )
);

// Store previous iteration values for under-relaxation
volScalarField rho_0
(
    IOobject
    (
        "rho_0",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    rho
);

volVectorField U_0
(
    IOobject
    (
        "U_0",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    U
);

volScalarField e_0
(
    IOobject
    (
        "e_0",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    thermo.he()
);

// Create the sonic limiter fields
volScalarField c("c", sqrt(thermo.Cp()/thermo.Cv() * thermo.gamma() * p/rho));
surfaceScalarField cSf("cSf", fvc::interpolate(c));

surfaceScalarField phiv
(
    IOobject
    (
        "phiv",
        runTime.timeName(),
        mesh
    ),
    mesh,
    dimensionedScalar(dimless, Zero)
);

// MULES flux from surf.phi
surfaceScalarField rhoPhiv
(
    IOobject
    (
        "rhoPhiv",
        runTime.timeName(),
        mesh
    ),
    mesh,
    dimensionedScalar(dimless, Zero)
);

// Needed for backward compatibility
dimensionedScalar v_zero(dimVolume/dimTime, Zero);

// Create temperature field for consistent thermodynamic evaluation
volScalarField mu
(
    IOobject
    (
        "mu",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    thermo.mu()
);

// Store the momentum to avoid calculating it multiple times
volVectorField rhoU
(
    IOobject
    (
        "rhoU",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    rho*U
);

// Cache grad(U) for viscous term
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

// Time control will be handled by standard OpenFOAM headers in main code
