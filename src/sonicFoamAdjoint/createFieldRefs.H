/*---------------------------------------------------------------------------*\
  Create references to fields for improved readability and access

  Sets up references to commonly accessed fields across multiple header files
  to improve consistency and avoid repetitive dereferences.
\*---------------------------------------------------------------------------*/

// Use reference to e (internal energy) from thermo
const volScalarField& e = thermo.he();

// Only set up references to adjoint fields if they exist
// These will be set after the adjoint fields are created
#ifdef CREATE_ADJOINT_REFERENCES
    volScalarField& psiRho = mesh.lookupObjectRef<volScalarField>("psi_rho");
    volVectorField& psiU = mesh.lookupObjectRef<volVectorField>("psi_U");
    volScalarField& psiE = mesh.lookupObjectRef<volScalarField>("psi_e");
    surfaceScalarField& psiPhi = mesh.lookupObjectRef<surfaceScalarField>("psi_phi");
#endif

// References to turbulence fields
#include "createRhoUf.H"
turbulence->validate();
