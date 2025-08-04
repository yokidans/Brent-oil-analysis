<#
.SYNOPSIS
    Ultimate solution with guaranteed path resolution
#>

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# 1. Set up debug directory
$debugDir = "$env:USERPROFILE\Desktop\oil_analysis_debug"
if (-not (Test-Path $debugDir)) {
    New-Item -ItemType Directory -Path $debugDir | Out-Null
}

# 2. Create Python script with absolute path handling
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$pythonScriptPath = "$debugDir\analysis_$timestamp.py"
$logPath = "$debugDir\output_$timestamp.log"

# Get absolute path to project root
$projectRoot = (Get-Item -Path ".\").FullName

@"
# -*- coding: utf-8 -*-
"""Ultimate path resolution script"""

import sys
import os
import traceback

def main():
    try:
        print("=== ENVIRONMENT CHECK ===")
        print(f"Python: {sys.version}")
        print(f"Executable: {sys.executable}")
        print(f"Working dir: {os.getcwd()}")

        # Absolute path to project root (from PowerShell)
        project_root = r"$projectRoot"
        print(f"\nProject root (from PS): {project_root}")

        # Verify project root exists
        if not os.path.exists(project_root):
            raise FileNotFoundError(f"Project root not found: {project_root}")

        # Add to Python path (first position)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        print("\n=== PYTHON PATH ===")
        for p in sys.path[:10]:  # Show first 10 paths
            print(f" - {p}")

        # Verify src/analysis exists
        src_path = os.path.join(project_root, 'src')
        analysis_path = os.path.join(src_path, 'analysis')
        print(f"\nChecking module path: {analysis_path}")
        print(f"Exists: {os.path.exists(analysis_path)}")
        if os.path.exists(analysis_path):
            print("Contents:", os.listdir(analysis_path))

        print("\n=== IMPORT TEST ===")
        try:
            import pandas as pd
            print(f"✓ Pandas {pd.__version__}")
        except Exception as e:
            print("✗ Pandas import failed!")
            traceback.print_exc()
            return 1

        try:
            # Absolute import using package structure
            from src.analysis.causal_analysis import CausalImpactAnalyzer
            print("✓ CausalImpactAnalyzer imported successfully!")
            return 0
        except Exception as e:
            print("✗ CausalImpactAnalyzer import failed!")
            traceback.print_exc()
            return 1

    except Exception as e:
        print("!!! UNEXPECTED ERROR !!!")
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
"@ | Out-File -FilePath $pythonScriptPath -Encoding utf8

# 3. Execute with proper environment
try {
    $pythonPath = "$projectRoot\venv\Scripts\python.exe"
    
    Write-Host "`n=== EXECUTION STARTED ===" -ForegroundColor Cyan
    Write-Host "Project root: $projectRoot" -ForegroundColor Yellow
    Write-Host "Python path: $pythonPath" -ForegroundColor Yellow
    Write-Host "Script: $pythonScriptPath" -ForegroundColor Yellow

    # Start process with explicit environment
    $processInfo = New-Object System.Diagnostics.ProcessStartInfo
    $processInfo.FileName = $pythonPath
    $processInfo.Arguments = $pythonScriptPath
    $processInfo.WorkingDirectory = $projectRoot
    $processInfo.RedirectStandardOutput = $true
    $processInfo.RedirectStandardError = $true
    $processInfo.UseShellExecute = $false
    $processInfo.CreateNoWindow = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $processInfo
    $process.Start() | Out-Null

    # Capture output streams
    $output = $process.StandardOutput.ReadToEnd()
    $errorOutput = $process.StandardError.ReadToEnd()
    $process.WaitForExit()

    # Save logs
    $output | Out-File -FilePath $logPath -Encoding utf8
    $errorOutput | Out-File -FilePath "$logPath.errors" -Encoding utf8

    # Display output
    Write-Host "`n=== PYTHON OUTPUT ===" -ForegroundColor Green
    Write-Host $output

    if ($errorOutput) {
        Write-Host "`n=== PYTHON ERRORS ===" -ForegroundColor Red
        Write-Host $errorOutput
    }

    if ($process.ExitCode -ne 0) {
        throw "Python script failed with exit code $($process.ExitCode)"
    }

    Write-Host "`n=== EXECUTION SUCCESSFUL ===" -ForegroundColor Green
}
catch {
    Write-Host "`n!!! EXECUTION FAILED !!!" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
finally {
    Write-Host "`nDebug files:" -ForegroundColor Cyan
    Write-Host "Script: $pythonScriptPath"
    Write-Host "Output: $logPath"
    Write-Host "Errors: $logPath.errors"
    Write-Host "`n=== COMPLETED ===" -ForegroundColor Cyan
}