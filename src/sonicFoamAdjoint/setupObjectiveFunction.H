// Setup the objective function for the adjoint solver

Info<< "Setting up objective function: " << objectiveName << nl << endl;

// Create a dictionary for objective function parameters
IOdictionary objectiveDict
(
    IOobject
    (
        "objectiveDict",
        runTime.constant(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::NO_WRITE
    )
);

// Check if the dictionary exists, if not, create default values
if (!objectiveDict.headerOk())
{
    Info<< "Objective function dictionary not found, creating default settings" << endl;
    
    objectiveDict.add("drag", dictionary());
    objectiveDict.subDict("drag").add("patches", wordList({"wall", "wing", "airfoil"}));
    objectiveDict.subDict("drag").add("direction", vector(1, 0, 0));
    objectiveDict.subDict("drag").add("scale", 1.0);
    
    objectiveDict.add("lift", dictionary());
    objectiveDict.subDict("lift").add("patches", wordList({"wall", "wing", "airfoil"}));
    objectiveDict.subDict("lift").add("direction", vector(0, 1, 0));
    objectiveDict.subDict("lift").add("scale", 1.0);
    
    objectiveDict.add("moment", dictionary());
    objectiveDict.subDict("moment").add("patches", wordList({"wall", "wing", "airfoil"}));
    objectiveDict.subDict("moment").add("direction", vector(0, 0, 1));
    objectiveDict.subDict("moment").add("center", vector(0, 0, 0));
    objectiveDict.subDict("moment").add("scale", 1.0);
    
    objectiveDict.add("pressure_loss", dictionary());
    objectiveDict.subDict("pressure_loss").add("inlet", wordList({"inlet"}));
    objectiveDict.subDict("pressure_loss").add("outlet", wordList({"outlet"}));
    objectiveDict.subDict("pressure_loss").add("scale", 1.0);
    
    objectiveDict.add("uniformity", dictionary());
    objectiveDict.subDict("uniformity").add("patches", wordList({"outlet"}));
    objectiveDict.subDict("uniformity").add("field", word("U"));
    objectiveDict.subDict("uniformity").add("scale", 1.0);
    
    objectiveDict.regIOobject::write();
}

// Check if the specified objective function exists
if (!objectiveDict.found(objectiveName))
{
    FatalErrorInFunction
        << "Objective function '" << objectiveName << "' not found in objectiveDict"
        << exit(FatalError);
}

// Create a reference to the objective dictionary
const dictionary& objDict = objectiveDict.subDict(objectiveName);

// Initialize the objective function value
scalar objectiveValue = 0.0;

// Lambda function to calculate the objective value
auto calculateObjective = [&]() -> scalar
{
    scalar obj = 0.0;
    
    if (objectiveName == "drag" || objectiveName == "lift")
    {
        // Get the force direction vector
        vector direction = objDict.get<vector>("direction");
        direction /= mag(direction); // Normalize
        
        // Get the list of patches to include
        wordList patchNames = objDict.get<wordList>("patches");
        
        // Calculate the force on the specified patches
        volTensorField gradU("gradU", fvc::grad(U));
        
        forAll(patchNames, i)
        {
            label patchI = mesh.boundaryMesh().findPatchID(patchNames[i]);
            
            if (patchI != -1)
            {
                const fvPatch& patch = mesh.boundary()[patchI];
                const vectorField& Sf = patch.Sf();
                
                // Pressure force component
                scalarField pForce(p.boundaryField()[patchI].size());
                forAll(pForce, faceI)
                {
                    pForce[faceI] = p.boundaryField()[patchI][faceI] * (Sf[faceI] & direction);
                }
                
                // Viscous force component
                vectorField viscousForce(Sf.size());
                forAll(viscousForce, faceI)
                {
                    viscousForce[faceI] = (
                        mu.boundaryField()[patchI][faceI]
                        * dev(gradU.boundaryField()[patchI][faceI])
                    ) & Sf[faceI];
                }
                
                scalarField vForce(viscousForce.size());
                forAll(vForce, faceI)
                {
                    vForce[faceI] = viscousForce[faceI] & direction;
                }
                
                // Total force
                obj += gSum(pForce + vForce);
            }
        }
    }
    else if (objectiveName == "moment")
    {
        // Get the moment axis direction
        vector direction = objDict.get<vector>("direction");
        direction /= mag(direction); // Normalize
        
        // Get the moment center
        vector center = objDict.get<vector>("center");
        
        // Get the list of patches to include
        wordList patchNames = objDict.get<wordList>("patches");
        
        // Calculate the moment on the specified patches
        volTensorField gradU("gradU", fvc::grad(U));
        
        forAll(patchNames, i)
        {
            label patchI = mesh.boundaryMesh().findPatchID(patchNames[i]);
            
            if (patchI != -1)
            {
                const fvPatch& patch = mesh.boundary()[patchI];
                const vectorField& Sf = patch.Sf();
                const vectorField& Cf = patch.Cf();
                
                // Lever arm
                vectorField r(Cf.size());
                forAll(r, faceI)
                {
                    r[faceI] = Cf[faceI] - center;
                }
                
                // Force
                vectorField pForce(Sf.size());
                forAll(pForce, faceI)
                {
                    pForce[faceI] = p.boundaryField()[patchI][faceI] * Sf[faceI];
                }
                
                vectorField viscousForce(Sf.size());
                forAll(viscousForce, faceI)
                {
                    viscousForce[faceI] = (
                        mu.boundaryField()[patchI][faceI]
                        * dev(gradU.boundaryField()[patchI][faceI])
                    ) & Sf[faceI];
                }
                
                vectorField totalForce(pForce.size());
                forAll(totalForce, faceI)
                {
                    totalForce[faceI] = pForce[faceI] + viscousForce[faceI];
                }
                
                // Moment = r × F
                forAll(r, faceI)
                {
                    obj += (r[faceI] ^ totalForce[faceI]) & direction;
                }
            }
        }
    }
    else if (objectiveName == "pressure_loss")
    {
        // Get the inlet and outlet patch names
        wordList inletNames = objDict.get<wordList>("inlet");
        wordList outletNames = objDict.get<wordList>("outlet");
        
        // Calculate the average pressure at inlet and outlet
        scalar inletPressure = 0.0;
        scalar inletArea = 0.0;
        scalar outletPressure = 0.0;
        scalar outletArea = 0.0;
        
        forAll(inletNames, i)
        {
            label patchI = mesh.boundaryMesh().findPatchID(inletNames[i]);
            
            if (patchI != -1)
            {
                const fvPatch& patch = mesh.boundary()[patchI];
                const scalarField& magSf = patch.magSf();
                
                inletPressure += gSum(p.boundaryField()[patchI] * magSf);
                inletArea += gSum(magSf);
            }
        }
        
        forAll(outletNames, i)
        {
            label patchI = mesh.boundaryMesh().findPatchID(outletNames[i]);
            
            if (patchI != -1)
            {
                const fvPatch& patch = mesh.boundary()[patchI];
                const scalarField& magSf = patch.magSf();
                
                outletPressure += gSum(p.boundaryField()[patchI] * magSf);
                outletArea += gSum(magSf);
            }
        }
        
        // Calculate the average pressures
        if (inletArea > SMALL && outletArea > SMALL)
        {
            scalar pInlet = inletPressure / inletArea;
            scalar pOutlet = outletPressure / outletArea;
            
            // Pressure loss objective
            obj = pInlet - pOutlet;
        }
    }
    else if (objectiveName == "uniformity")
    {
        // Get the patch names
        wordList patchNames = objDict.get<wordList>("patches");
        word fieldName = objDict.get<word>("field");
        
        // Check which field to use
        if (fieldName == "U")
        {
            forAll(patchNames, i)
            {
                label patchI = mesh.boundaryMesh().findPatchID(patchNames[i]);
                
                if (patchI != -1)
                {
                    const fvPatch& patch = mesh.boundary()[patchI];
                    const scalarField& magSf = patch.magSf();
                    
                    // Get the field
                    const vectorField& Ub = U.boundaryField()[patchI];
                    
                    // Calculate average velocity
                    vector Uavg = gSum(Ub * magSf) / gSum(magSf);
                    
                    // Calculate deviation from average
                    forAll(Ub, faceI)
                    {
                        obj += magSf[faceI] * magSqr(Ub[faceI] - Uavg);
                    }
                    
                    // Normalize by total area
                    obj /= gSum(magSf);
                }
            }
        }
        else if (fieldName == "p")
        {
            forAll(patchNames, i)
            {
                label patchI = mesh.boundaryMesh().findPatchID(patchNames[i]);
                
                if (patchI != -1)
                {
                    const fvPatch& patch = mesh.boundary()[patchI];
                    const scalarField& magSf = patch.magSf();
                    
                    // Get the field
                    const scalarField& pb = p.boundaryField()[patchI];
                    
                    // Calculate average pressure
                    scalar pavg = gSum(pb * magSf) / gSum(magSf);
                    
                    // Calculate deviation from average
                    forAll(pb, faceI)
                    {
                        obj += magSf[faceI] * sqr(pb[faceI] - pavg);
                    }
                    
                    // Normalize by total area
                    obj /= gSum(magSf);
                }
            }
        }
    }
    
    // Apply scaling factor
    scalar scale = objDict.getOrDefault<scalar>("scale", 1.0);
    obj *= scale;
    
    return obj;
};

// Calculate the initial objective value
objectiveValue = calculateObjective();

Info<< "Initial objective value (" << objectiveName << "): " << objectiveValue << nl << endl;

// Create a directory to store objective history
{
    fileName objDir = runTime.path()/"objective";
    if (!isDir(objDir))
    {
        mkDir(objDir);
    }
}

// Create file to store objective history
OFstream objectiveFile(runTime.path()/"objective"/objectiveName);
objectiveFile << "# Time\t" << objectiveName << nl;
objectiveFile << runTime.timeName() << "\t" << objectiveValue << nl;
