// Create adjoint fields
Info<< "Creating adjoint fields\n" << endl;

// Adjoint field for continuity equation (density)
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
    dimensionedScalar("zero", dimless, 0.0)
);

// Adjoint field for momentum equation (velocity)
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
    dimensionedVector("zero", dimless, vector::zero)
);

// Adjoint field for energy equation
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
    dimensionedScalar("zero", dimless, 0.0)
);

// Create sensitivity field (will store the computed sensitivities)
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
    dimensionedVector("zero", dimensionSet(1, -1, -2, 0, 0, 0, 0), vector::zero)
);

// Optional field for adjoint turbulence variables
// For now, we're using a frozen turbulence approach where
// turbulence variables are not included in the adjoint system
volScalarField psi_nut
(
    IOobject
    (
        "psi_nut",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("zero", dimless, 0.0)
);

// Create optimization parameters
dictionary* objFuncDict = nullptr;
if (optimizationDict.found("objectiveFunction"))
{
    objFuncDict = &optimizationDict.subDict("objectiveFunction");
}
else
{
    Info<< "Warning: objectiveFunction dictionary not found in optimizationDict."
        << " Using default values." << endl;
}

// Default values for objective function
wordList patchNames;
if (objFuncDict && objFuncDict->found("patches"))
{
    patchNames = objFuncDict->lookup("patches");
}
else
{
    // Try to find wall patches as default
    const polyBoundaryMesh& patches = mesh.boundaryMesh();
    forAll(patches, patchI)
    {
        const polyPatch& pp = patches[patchI];
        if (pp.type() == "wall")
        {
            patchNames.append(pp.name());
        }
    }
    
    if (patchNames.size() == 0)
    {
        FatalErrorInFunction
            << "No patches specified for objective function and no wall patches found."
            << exit(FatalError);
    }
}

// Force direction for drag/lift computation
vector forceDirection(1, 0, 0); // Default: drag in x-direction
if (objFuncDict && objFuncDict->found("direction"))
{
    forceDirection = vector(objFuncDict->lookup("direction"));
    forceDirection /= mag(forceDirection); // Normalize
}

// Scale factor for objective
scalar objFuncScale = 1.0;
if (objFuncDict && objFuncDict->found("scale"))
{
    objFuncScale = readScalar(objFuncDict->lookup("scale"));
}

// Reference values for normalization
scalar rhoRef = 1.0;
scalar URef = 1.0;
scalar LRef = 1.0;
scalar pRef = 1.0;

if (objFuncDict)
{
    if (objFuncDict->found("rhoRef")) rhoRef = readScalar(objFuncDict->lookup("rhoRef"));
    if (objFuncDict->found("URef")) URef = readScalar(objFuncDict->lookup("URef"));
    if (objFuncDict->found("LRef")) LRef = readScalar(objFuncDict->lookup("LRef"));
    if (objFuncDict->found("pRef")) pRef = readScalar(objFuncDict->lookup("pRef"));
}

// Store objective function value for convergence monitoring
scalar objectiveValue = 0.0;
