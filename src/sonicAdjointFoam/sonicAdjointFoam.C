/*---------------------------------------------------------------------------*\
    sonicAdjointFoam: Adjoint solver for supersonic compressible flows
    with free-form deformation (FFD) integration

    This solver is designed to work with OpenFFD for sensitivity-based 
    shape optimization of supersonic flow applications
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "fluidThermo.H"
#include "turbulentFluidThermoModel.H"
#include "fvOptions.H"
#include "pimpleControl.H"
#include "CorrectPhi.H"
#include "fixedFluxPressureFvPatchScalarField.H"
#include "localEulerDdtScheme.H"
#include "fvcSmooth.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Adjoint solver for supersonic compressible flows"
        " with OpenFFD integration."
    );

    #include "postProcess.H"
    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createControl.H"
    #include "createFields.H"
    #include "globalVariables.H"
    #include "createFieldRefs.H"
    #include "createTimeControls.H"
    #include "createFvOptions.H"
    #include "initContinuityErrs.H"
    #include "createAdjointFields.H"

    turbulence->validate();

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    
    Info<< "\nStarting time loop\n" << endl;

    // Run forward simulation first to get the baseline solution
    #include "runForwardSolution.H"

    // Run adjoint simulation to calculate sensitivities
    #include "solveAdjoint.H"

    // Calculate sensitivities and prepare for FFD integration
    #include "calculateSensitivities.H"

    // Export sensitivities in a format compatible with OpenFFD
    #include "exportSensitivities.H"

    Info<< "End\n" << endl;

    return 0;
}

// ************************************************************************* //
