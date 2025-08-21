#!/bin/bash

echo "🚀 COMPLETE PARAMETER OPTIMIZATION SUITE"
echo "This will run all parameter sweeps and analysis"
echo "Estimated time: 15-30 minutes depending on hardware"
echo ""

# Make scripts executable
chmod +x parameter_sweep_beta.sh
chmod +x parameter_sweep_threshold.sh  
chmod +x parameter_sweep_layers.sh
chmod +x analyze_sweep_results.py

echo "📊 Step 1/4: Beta Weight Grid Search (27 combinations)"
echo "Testing β ∈ {0.05, 0.1, 0.2} for templ_frob, templ_band, templ_eig..."
./parameter_sweep_beta.sh

echo ""
echo "🎯 Step 2/4: Threshold Optimization (10 thresholds)"
echo "Testing flip thresholds from 0.3 to 0.8..."
./parameter_sweep_threshold.sh

echo ""
echo "📊 Step 3/4: Template Layer Count Optimization (5 layer counts)"
echo "Testing 4, 6, 8, 10, 12 layers in correlation template..."
./parameter_sweep_layers.sh

echo ""
echo "🔍 Step 4/4: Results Analysis"
echo "Analyzing all sweep results and finding optimal parameters..."
python analyze_sweep_results.py

echo ""
echo "✅ COMPLETE PARAMETER OPTIMIZATION FINISHED!"
echo ""
echo "📁 Generated files:"
echo "  - parameter_sweep_results.csv (beta weights)"
echo "  - threshold_sweep_results.csv (flip thresholds)" 
echo "  - layer_sweep_results.csv (template layers)"
echo "  - correlation_template_*L.json (different layer templates)"
echo ""
echo "🎯 Check the analysis output above for optimal parameter recommendations!"