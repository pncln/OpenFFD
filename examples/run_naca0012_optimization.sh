#!/bin/bash

# NACA0012 Airfoil Optimization with OpenFOAM
# 
# This script demonstrates how to run the complete NACA0012 optimization workflow
# 

echo "🎯 NACA0012 Airfoil Shape Optimization with OpenFOAM"
echo "============================================================"

# Check if OpenFOAM is available
if ! command -v openfoam &> /dev/null; then
    echo "❌ OpenFOAM not found!"
    echo "Please install OpenFOAM or ensure it's in your PATH"
    exit 1
fi

echo "✅ OpenFOAM found"

# Activate OpenFOAM environment and run optimization
echo "🔧 Activating OpenFOAM environment and running optimization..."
echo ""

# Method 1: Run in activated environment
openfoam python3 naca0012_optimization.py

# Check if optimization was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 NACA0012 optimization completed successfully!"
    echo ""
    echo "📁 Generated files:"
    echo "   - naca0012_optimization/     (Complete OpenFOAM case)"
    echo "   - naca0012_optimization.log  (Detailed execution log)"
    echo "   - optimization_results.txt   (Final results summary)"
    echo ""
    echo "📊 To view results:"
    echo "   cat naca0012_optimization/optimization_results.txt"
    echo ""
    echo "🔍 To inspect the case:"
    echo "   cd naca0012_optimization"
    echo "   paraFoam"
else
    echo ""
    echo "❌ Optimization failed. Check the log file for details:"
    echo "   tail -50 naca0012_optimization.log"
fi