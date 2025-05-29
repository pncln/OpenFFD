/*---------------------------------------------------------------------------*\
  Optimization control and algorithm integration

  Implements interfaces to various optimization algorithms and provides
  a framework for controlling the optimization loop.
\*---------------------------------------------------------------------------*/

class OptimizationController
{
private:
    const Time& runTime_;
    const fvMesh& mesh_;
    const word objectiveName_;
    
    // Optimization parameters
    word algorithm_;
    label maxIterations_;
    scalar convergenceTol_;
    scalar stepSize_;
    scalar initialStepSize_;
    scalar stepSizeDecay_;
    label adaptInterval_;
    scalar wolfeC1_;
    scalar wolfeC2_;
    
    // Optimization state
    label iteration_;
    scalarList objectiveHistory_;
    List<vectorField> gradientHistory_;
    List<vectorField> designVarHistory_;
    
    // Current design variables and gradient
    vectorField designVariables_;
    vectorField gradient_;
    
    // Line search parameters
    scalar currentStepSize_;
    vectorField searchDirection_;
    
public:
    //- Constructor
    OptimizationController
    (
        const Time& runTime,
        const fvMesh& mesh,
        const word& objectiveName
    )
    :
        runTime_(runTime),
        mesh_(mesh),
        objectiveName_(objectiveName),
        algorithm_("gradientDescent"),
        maxIterations_(100),
        convergenceTol_(1e-5),
        stepSize_(0.01),
        initialStepSize_(0.01),
        stepSizeDecay_(0.9),
        adaptInterval_(5),
        wolfeC1_(1e-4),
        wolfeC2_(0.9),
        iteration_(0),
        currentStepSize_(0.01)
    {
        // Read configuration from dictionary if available
        IOdictionary optimizationDict
        (
            IOobject
            (
                "optimizationDict",
                runTime.constant(),
                mesh,
                IOobject::READ_IF_PRESENT,
                IOobject::NO_WRITE
            )
        );
        
        if (optimizationDict.headerOk())
        {
            algorithm_ = optimizationDict.getOrDefault<word>("algorithm", "gradientDescent");
            maxIterations_ = optimizationDict.getOrDefault<label>("maxIterations", 100);
            convergenceTol_ = optimizationDict.getOrDefault<scalar>("convergenceTol", 1e-5);
            stepSize_ = optimizationDict.getOrDefault<scalar>("stepSize", 0.01);
            initialStepSize_ = optimizationDict.getOrDefault<scalar>("initialStepSize", 0.01);
            stepSizeDecay_ = optimizationDict.getOrDefault<scalar>("stepSizeDecay", 0.9);
            adaptInterval_ = optimizationDict.getOrDefault<label>("adaptInterval", 5);
            wolfeC1_ = optimizationDict.getOrDefault<scalar>("wolfeC1", 1e-4);
            wolfeC2_ = optimizationDict.getOrDefault<scalar>("wolfeC2", 0.9);
        }
        
        currentStepSize_ = initialStepSize_;
        
        Info<< "Initialized optimization controller:" << nl
            << "  Algorithm: " << algorithm_ << nl
            << "  Max iterations: " << maxIterations_ << nl
            << "  Convergence tolerance: " << convergenceTol_ << nl
            << "  Step size: " << stepSize_ << nl
            << "  Initial step size: " << initialStepSize_ << nl
            << "  Step size decay: " << stepSizeDecay_ << nl
            << "  Adaptation interval: " << adaptInterval_ << nl;
        
        if (algorithm_ == "lineSearch")
        {
            Info<< "  Wolfe conditions: c1=" << wolfeC1_ << ", c2=" << wolfeC2_ << nl;
        }
        
        Info<< endl;
        
        // Create directory for optimization history
        if (!Foam::isDir(runTime_.path() + "/optimization"))
        {
            mkDir(runTime_.path()/"optimization");
        }
        
        // Create file for optimization history
        OFstream historyFile(runTime_.path()/"optimization"/"history.csv");
        historyFile << "Iteration,Objective,StepSize,GradientNorm" << nl;
    }
    
    //- Destructor
    ~OptimizationController()
    {}
    
    //- Initialize design variables from sensitivities on surfaces
    void initializeFromSensitivities(const volVectorField& sensitivity)
    {
        // Get boundary mesh
        const polyBoundaryMesh& boundaryMesh = mesh_.boundaryMesh();
        
        // Count total boundary points for design variables
        label nDesignVars = 0;
        
        // First pass: count design variables
        forAll(boundaryMesh, patchI)
        {
            const polyPatch& patch = boundaryMesh[patchI];
            const word& patchName = patch.name();
            
            // Check if this patch is part of the design surface
            bool isDesignSurface = false;
            
            // For drag/lift/moment objectives, use wall patches
            if ((objectiveName_ == "drag" || objectiveName_ == "lift" || objectiveName_ == "moment") && 
                (patchName.find("wall") != string::npos || 
                 patchName.find("Wing") != string::npos || 
                 patchName.find("wing") != string::npos || 
                 patchName.find("Airfoil") != string::npos || 
                 patchName.find("airfoil") != string::npos))
            {
                isDesignSurface = true;
            }
            
            if (isDesignSurface)
            {
                nDesignVars += patch.nPoints();
            }
        }
        
        // Initialize design variables and gradient arrays
        designVariables_.setSize(nDesignVars);
        gradient_.setSize(nDesignVars);
        searchDirection_.setSize(nDesignVars);
        
        // Second pass: initialize design variables
        label varI = 0;
        forAll(boundaryMesh, patchI)
        {
            const polyPatch& patch = boundaryMesh[patchI];
            const word& patchName = patch.name();
            
            // Check if this patch is part of the design surface
            bool isDesignSurface = false;
            
            // For drag/lift/moment objectives, use wall patches
            if ((objectiveName_ == "drag" || objectiveName_ == "lift" || objectiveName_ == "moment") && 
                (patchName.find("wall") != string::npos || 
                 patchName.find("Wing") != string::npos || 
                 patchName.find("wing") != string::npos || 
                 patchName.find("Airfoil") != string::npos || 
                 patchName.find("airfoil") != string::npos))
            {
                isDesignSurface = true;
            }
            
            if (isDesignSurface)
            {
                // Get patch mesh points
                const labelList& meshPoints = patch.meshPoints();
                
                // Get patch sensitivities
                const vectorField& patchSens = sensitivity.boundaryField()[patchI];
                
                // Map face sensitivities to points
                Map<vector> pointSensMap;
                Map<label> pointCountMap;
                
                // First, accumulate sensitivities for each point
                forAll(patch, faceI)
                {
                    const face& f = patch[faceI];
                    const vector& faceSens = patchSens[faceI];
                    
                    forAll(f, facePointI)
                    {
                        label pointI = f[facePointI];
                        label meshPointI = patch.meshPoints()[pointI];
                        
                        if (!pointSensMap.found(meshPointI))
                        {
                            pointSensMap.insert(meshPointI, faceSens);
                            pointCountMap.insert(meshPointI, 1);
                        }
                        else
                        {
                            pointSensMap[meshPointI] += faceSens;
                            pointCountMap[meshPointI]++;
                        }
                    }
                }
                
                // Then, compute the average sensitivity for each point
                forAll(meshPoints, pointI)
                {
                    label meshPointI = meshPoints[pointI];
                    
                    if (pointSensMap.found(meshPointI))
                    {
                        vector pointSens = pointSensMap[meshPointI] / pointCountMap[meshPointI];
                        
                        // Initialize design variable to zero (current shape)
                        designVariables_[varI] = vector::zero;
                        
                        // Initialize gradient to the sensitivity
                        gradient_[varI] = pointSens;
                        
                        varI++;
                    }
                }
            }
        }
        
        Info<< "Initialized " << nDesignVars << " design variables from surface sensitivities" << nl
            << "  Gradient L2 norm: " << Foam::sqrt(gSum(magSqr(gradient_))/gradient_.size()) << nl << endl;
    }
    
    //- Update design variables using the chosen optimization algorithm
    bool updateDesignVariables()
    {
        if (algorithm_ == "gradientDescent")
        {
            // Standard gradient descent
            forAll(designVariables_, i)
            {
                designVariables_[i] -= stepSize_ * gradient_[i];
            }
        }
        else if (algorithm_ == "momentum")
        {
            // Gradient descent with momentum
            scalar beta = 0.9;  // Momentum coefficient
            
            if (iteration_ == 0 || gradientHistory_.size() == 0)
            {
                // First iteration: standard gradient descent
                forAll(designVariables_, i)
                {
                    designVariables_[i] -= stepSize_ * gradient_[i];
                }
            }
            else
            {
                // Use momentum from previous iteration
                const vectorField& prevGradient = gradientHistory_[gradientHistory_.size() - 1];
                
                forAll(designVariables_, i)
                {
                    designVariables_[i] -= stepSize_ * gradient_[i] + beta * prevGradient[i];
                }
            }
        }
        else if (algorithm_ == "adam")
        {
            // Adam optimizer
            scalar beta1 = 0.9;   // Exponential decay for first moment
            scalar beta2 = 0.999; // Exponential decay for second moment
            scalar epsilon = 1e-8; // Small constant for numerical stability
            
            // Initialize moment estimates if first iteration
            static vectorField m, v;
            if (iteration_ == 0 || gradientHistory_.size() == 0)
            {
                m = vectorField(gradient_.size(), vector::zero);
                v = vectorField(gradient_.size(), vector::zero);
            }
            
            // Update biased first and second moment estimates
            forAll(gradient_, i)
            {
                m[i] = beta1 * m[i] + (1.0 - beta1) * gradient_[i];
                v[i] = beta2 * v[i] + (1.0 - beta2) * cmptMultiply(gradient_[i], gradient_[i]);
            }
            
            // Compute bias-corrected estimates
            scalar corrFactor1 = 1.0 / (1.0 - pow(beta1, iteration_ + 1));
            scalar corrFactor2 = 1.0 / (1.0 - pow(beta2, iteration_ + 1));
            
            // Update design variables
            forAll(designVariables_, i)
            {
                vector mCorrected = m[i] * corrFactor1;
                vector vCorrected = v[i] * corrFactor2;
                
                // Element-wise division and square root for v
                vector denominator(
                    Foam::sqrt(vCorrected.x()) + epsilon,
                    Foam::sqrt(vCorrected.y()) + epsilon,
                    Foam::sqrt(vCorrected.z()) + epsilon
                );
                
                // Element-wise division
                vector update(
                    mCorrected.x() / denominator.x(),
                    mCorrected.y() / denominator.y(),
                    mCorrected.z() / denominator.z()
                );
                
                designVariables_[i] -= stepSize_ * update;
            }
        }
        else if (algorithm_ == "lineSearch")
        {
            // Steepest descent with line search
            // This is a simplified implementation of a backtracking line search
            // In a real implementation, more sophisticated line search methods would be used
            
            // Compute the search direction (negative gradient)
            searchDirection_ = -1.0 * gradient_;
            
            // Perform line search to find optimal step size
            performLineSearch();
            
            // Update design variables
            forAll(designVariables_, i)
            {
                designVariables_[i] += currentStepSize_ * searchDirection_[i];
            }
        }
        else if (algorithm_ == "lbfgs")
        {
            // L-BFGS (Limited-memory BFGS) algorithm
            // This is a simplified placeholder for L-BFGS
            // In a real implementation, a proper L-BFGS solver would be used
            
            // Maximum number of past iterations to store
            
            // If we don't have enough history, use gradient descent
            if (iteration_ < 2 || gradientHistory_.size() < 2 || designVarHistory_.size() < 2)
            {
                forAll(designVariables_, i)
                {
                    designVariables_[i] -= stepSize_ * gradient_[i];
                }
            }
            else
            {
                // Compute L-BFGS direction
                // This is a placeholder for a real L-BFGS implementation
                // It would use stored gradients and design variable differences
                // to approximate the inverse Hessian
                
                // In this simplified version, we'll just use the negative gradient
                // with a scaled step size based on iteration
                scalar scaleFactor = 1.0 + 0.1 * min(iteration_, 10);
                
                forAll(designVariables_, i)
                {
                    designVariables_[i] -= scaleFactor * stepSize_ * gradient_[i];
                }
            }
        }
        
        // Store history
        objectiveHistory_.append(computeObjectiveValue());
        gradientHistory_.append(gradient_);
        designVarHistory_.append(designVariables_);
        
        // Adapt step size if needed
        if (adaptInterval_ > 0 && iteration_ % adaptInterval_ == 0 && iteration_ > 0)
        {
            adaptStepSize();
        }
        
        // Update iteration counter
        iteration_++;
        
        // Write optimization history
        OFstream historyFile(runTime_.path()/"optimization"/"history.csv", IOstreamOption(IOstreamOption::APPEND));
        historyFile << iteration_ << ","
                   << objectiveHistory_[objectiveHistory_.size() - 1] << ","
                   << currentStepSize_ << ","
                   << Foam::sqrt(gSum(magSqr(gradient_))/gradient_.size()) << nl;
        
        // Check for convergence
        if (checkConvergence())
        {
            Info<< "Optimization converged after " << iteration_ << " iterations" << nl << endl;
            return false;
        }
        
        // Check for maximum iterations
        if (iteration_ >= maxIterations_)
        {
            Info<< "Reached maximum number of iterations: " << maxIterations_ << nl << endl;
            return false;
        }
        
        return true;
    }
    
    //- Apply the current design variables to the mesh
    bool applyDesignVariables(pointField& points) const
    {
        // Get boundary mesh
        const polyBoundaryMesh& boundaryMesh = mesh_.boundaryMesh();
        
        // Design variable index counter
        label varI = 0;
        
        // Apply design variables to mesh points
        forAll(boundaryMesh, patchI)
        {
            const polyPatch& patch = boundaryMesh[patchI];
            const word& patchName = patch.name();
            
            // Check if this patch is part of the design surface
            bool isDesignSurface = false;
            
            // For drag/lift/moment objectives, use wall patches
            if ((objectiveName_ == "drag" || objectiveName_ == "lift" || objectiveName_ == "moment") && 
                (patchName.find("wall") != string::npos || 
                 patchName.find("Wing") != string::npos || 
                 patchName.find("wing") != string::npos || 
                 patchName.find("Airfoil") != string::npos || 
                 patchName.find("airfoil") != string::npos))
            {
                isDesignSurface = true;
            }
            
            if (isDesignSurface)
            {
                // Get patch mesh points
                const labelList& meshPoints = patch.meshPoints();
                
                // Apply design variables to mesh points
                forAll(meshPoints, pointI)
                {
                    label meshPointI = meshPoints[pointI];
                    
                    if (varI < designVariables_.size())
                    {
                        points[meshPointI] += designVariables_[varI];
                        varI++;
                    }
                }
            }
        }
        
        return true;
    }
    
    //- Get the current iteration
    label iteration() const
    {
        return iteration_;
    }
    
    //- Get the current step size
    scalar stepSize() const
    {
        return currentStepSize_;
    }
    
    //- Set the current objective value
    void setObjectiveValue(const scalar& value)
    {
        if (objectiveHistory_.size() == 0)
        {
            objectiveHistory_.append(value);
        }
        else
        {
            objectiveHistory_[objectiveHistory_.size() - 1] = value;
        }
    }
    
    //- Set the current gradient
    void setGradient(const vectorField& grad)
    {
        gradient_ = grad;
    }
    
private:
    //- Compute the objective value for the current design
    scalar computeObjectiveValue() const
    {
        // In a real implementation, this would evaluate the objective function
        // Here, we'll return the last stored value or a placeholder
        if (objectiveHistory_.size() > 0)
        {
            return objectiveHistory_[objectiveHistory_.size() - 1];
        }
        
        return 0.0;
    }
    
    //- Check for convergence
    bool checkConvergence() const
    {
        // Check gradient norm
        scalar gradNorm = Foam::sqrt(gSum(magSqr(gradient_))/gradient_.size());
        
        if (gradNorm < convergenceTol_)
        {
            return true;
        }
        
        // Check objective value change if we have enough history
        if (objectiveHistory_.size() >= 2)
        {
            scalar prevObj = objectiveHistory_[objectiveHistory_.size() - 2];
            scalar currObj = objectiveHistory_[objectiveHistory_.size() - 1];
            
            scalar relChange = mag(currObj - prevObj) / (mag(prevObj) + SMALL);
            
            if (relChange < convergenceTol_)
            {
                return true;
            }
        }
        
        return false;
    }
    
    //- Adapt the step size based on recent history
    void adaptStepSize()
    {
        // Simple step size decay
        currentStepSize_ *= stepSizeDecay_;
        
        Info<< "Adapted step size to " << currentStepSize_ << nl << endl;
    }
    
    //- Perform a line search to find optimal step size
    void performLineSearch()
    {
        // Initialize step size
        currentStepSize_ = initialStepSize_;
        
        // Maximum number of line search iterations
        label maxLineSearchIters = 10;
        
        // Current objective value
        scalar f0 = computeObjectiveValue();
        
        // Directional derivative
        scalar gradDotDir = sum(gradient_ & searchDirection_);
        
        // Backtracking line search
        for (label i = 0; i < maxLineSearchIters; i++)
        {
            // Test step size
            // In a real implementation, this would update the mesh and evaluate the objective
            // Here we just use a simplified model
            scalar fTest = f0 + 0.5 * currentStepSize_ * gradDotDir;
            
            // Check Armijo condition (sufficient decrease)
            if (fTest <= f0 + wolfeC1_ * currentStepSize_ * gradDotDir)
            {
                // Step size is acceptable
                break;
            }
            
            // Decrease step size
            currentStepSize_ *= 0.5;
        }
        
        Info<< "Line search determined step size: " << currentStepSize_ << nl << endl;
    }
};

// Create optimization controller
OptimizationController optimizationController(runTime, mesh, objectiveName);

// Example usage:
// 1. Initialize from sensitivities:
//    optimizationController.initializeFromSensitivities(meshSensitivity);
//
// 2. Run optimization loop:
//    while (optimizationController.updateDesignVariables())
//    {
//        // Apply design variables to mesh
//        pointField newPoints = mesh.points();
//        optimizationController.applyDesignVariables(newPoints);
//        mesh.movePoints(newPoints);
//        
//        // Run flow solver and adjoint
//        // ...
//        
//        // Update gradient and objective
//        optimizationController.setObjectiveValue(objectiveValue);
//        optimizationController.setGradient(convertedGradient);
//    }
