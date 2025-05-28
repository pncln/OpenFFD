/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2024 OpenFFD Project
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    sonicFoamAdjoint

Description
    Adjoint solver for transonic/supersonic flow of a compressible gas.
    Uses the PIMPLE (PISO-SIMPLE) algorithm to solve the time-stepping equations.
    Includes adjoint formulation for sensitivity calculations.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "psiThermo.H"
#include "turbulentFluidThermoModel.H"
#include "pimpleControl.H"
#include "fvOptions.H"
#include "localEulerDdtScheme.H"
#include "fvcSmooth.H"
#include "mathematicalConstants.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Adjoint solver for transonic/supersonic flow of a compressible gas."
    );

    argList::addOption
    (
        "objective",
        "name",
        "Specify the objective function (drag, lift, etc.)"
    );

    #include "postProcess.H"

    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createControl.H"
    #include "createFields.H"
    #include "createFieldRefs.H"
    #include "createFvOptions.H"
    #include "createTimeControls.H"
    #include "initContinuityErrs.H"
    
    // Read the objective function name
    word objectiveName = args.getOrDefault<word>("objective", "drag");
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    
    Info<< "\nStarting time loop\n" << endl;
    
    // Run the forward solution first (if needed)
    // This ensures we have the flow field to use for the adjoint equations
    #include "runForwardSolution.H"
    
    // Initialize adjoint variables
    #include "createAdjointFields.H"
    
    // Setup the objective function
    #include "setupObjectiveFunction.H"
    
    // Run the adjoint solution
    #include "solveAdjoint.H"
    
    // Calculate sensitivities
    #include "calculateSensitivities.H"
    
    // Export sensitivities for use by external tools (e.g., OpenFFD)
    #include "exportSensitivities.H"
    
    Info<< "End\n" << endl;
    
    return 0;
}

// ************************************************************************* //
