// Global variables for optimization control
word objectiveType("drag");
word sensFormat("OpenFFD");
scalar stepSize = 0.01;
bool smoothSensitivity = true;

// Check if we have a optimization dictionary and load values
if (optimizationDict.headerOk())
{
    objectiveType = optimizationDict.lookupOrDefault<word>("objectiveType", "drag");
    sensFormat = optimizationDict.lookupOrDefault<word>("sensitivityFormat", "OpenFFD");
    stepSize = optimizationDict.lookupOrDefault<scalar>("stepSize", 0.01);
    smoothSensitivity = optimizationDict.lookupOrDefault<bool>("smoothSensitivity", true);
}
