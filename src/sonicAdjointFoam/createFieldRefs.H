// Reference to the thermophysical properties for easier access
const volScalarField& psi = thermo.psi();
const volScalarField& mu = thermo.mu();
const volScalarField& alpha = thermo.alpha();
const volScalarField& Cp = thermo.Cp();
const volScalarField& Cv = thermo.Cv();
const scalar gamma = Cp[0]/Cv[0]; // Assuming constant gamma

// References to turbulence fields
const volScalarField& alphat = turbulence->alphat();
const volScalarField& mut = turbulence->mut();
const volScalarField& muEff = turbulence->muEff();
const volScalarField& alphaEff = turbulence->alphaEff();
