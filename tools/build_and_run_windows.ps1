param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $AppArgs
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'last_run.log'
$fatal = Join-Path $logDir 'last_fatal.txt'
Remove-Item $log -Force -ErrorAction SilentlyContinue
Remove-Item $fatal -Force -ErrorAction SilentlyContinue

Start-Transcript -Path $log -Force | Out-Null
$exitCode = 1
try {
    Write-Host '=== Vulkan Hair Studio launcher ==='
    Write-Host ('Project: ' + $root)
    Write-Host ('Time:    ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'))
    Write-Host ''

    if ([string]::IsNullOrWhiteSpace($env:VULKAN_SDK)) {
        throw 'VULKAN_SDK is not set. Install/reopen the LunarG Vulkan SDK environment before building.'
    }
    $glslc = Join-Path $env:VULKAN_SDK 'Bin\glslc.exe'
    if (-not (Test-Path $glslc)) {
        $glslc = Join-Path $env:VULKAN_SDK 'bin\glslc.exe'
    }
    if (-not (Test-Path $glslc)) {
        throw ('glslc.exe was not found under VULKAN_SDK: ' + $env:VULKAN_SDK)
    }
    $cmakeCmd = Get-Command cmake -ErrorAction SilentlyContinue
    if (-not $cmakeCmd) {
        throw 'CMake was not found in PATH.'
    }
    $cmake = $cmakeCmd.Source

    Write-Host ('VULKAN_SDK: ' + $env:VULKAN_SDK)
    Write-Host ('glslc:     ' + $glslc)
    Write-Host ('cmake:     ' + $cmake)
    & $cmake --version
    Write-Host ''

    if ($env:V47_PREFLIGHT -eq '1') {
        $python = Get-Command py -ErrorAction SilentlyContinue
        if ($python) {
            Write-Host '--- source/asset preflight ---'
            & $python.Source -3 -u verify_project.py
            if ($LASTEXITCODE -ne 0) { throw ('verify_project.py failed with exit code ' + $LASTEXITCODE) }
            & $python.Source -3 -u tools/validate_groom_assets.py
            if ($LASTEXITCODE -ne 0) { throw ('validate_groom_assets.py failed with exit code ' + $LASTEXITCODE) }
            & $python.Source -3 -u tools/test_v47_transport.py
            if ($LASTEXITCODE -ne 0) { throw ('test_v47_transport.py failed with exit code ' + $LASTEXITCODE) }
        } else {
            $python = Get-Command python -ErrorAction SilentlyContinue
            if ($python) {
                Write-Host '--- source/asset preflight ---'
                & $python.Source -u verify_project.py
                if ($LASTEXITCODE -ne 0) { throw ('verify_project.py failed with exit code ' + $LASTEXITCODE) }
                & $python.Source -u tools/validate_groom_assets.py
                if ($LASTEXITCODE -ne 0) { throw ('validate_groom_assets.py failed with exit code ' + $LASTEXITCODE) }
                & $python.Source -u tools/test_v47_transport.py
                if ($LASTEXITCODE -ne 0) { throw ('test_v47_transport.py failed with exit code ' + $LASTEXITCODE) }
            } else {
                Write-Warning 'V47_PREFLIGHT=1, but Python 3 was not found; preflight skipped.'
            }
        }
        Write-Host ''
    }

    if ($env:V47_CLEAN -eq '1' -and (Test-Path 'build')) {
        Write-Host '--- clean build requested ---'
        Remove-Item -Recurse -Force 'build'
    }

    Write-Host '--- CMake configure ---'
    & $cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    if ($LASTEXITCODE -ne 0) { throw ('CMake configure failed with exit code ' + $LASTEXITCODE) }

    Write-Host ''
    Write-Host '--- CMake build (Release) ---'
    & $cmake --build build --config Release --parallel
    if ($LASTEXITCODE -ne 0) { throw ('CMake build failed with exit code ' + $LASTEXITCODE) }

    # Do not assume a multi-config Visual Studio layout. CMake may legitimately
    # use Ninja/NMake and place the executable directly under build/.
    $candidates = @(
        (Join-Path $root 'build\Release\vulkan_hair_v47.exe'),
        (Join-Path $root 'build\vulkan_hair_v47.exe'),
        (Join-Path $root 'build\RelWithDebInfo\vulkan_hair_v47.exe')
    )
    $exe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $exe) {
        $found = Get-ChildItem -Path (Join-Path $root 'build') -Filter 'vulkan_hair_v47.exe' -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) { $exe = $found.FullName }
    }
    if (-not $exe) {
        throw 'Build reported success, but vulkan_hair_v47.exe could not be located under build/.'
    }

    Write-Host ''
    Write-Host ('--- Launching: ' + $exe + ' ---')
    if ($AppArgs) { Write-Host ('Args: ' + ($AppArgs -join ' ')) }
    & $exe @AppArgs
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        if ($exitCode -eq -1073741571) {
            throw 'vulkan_hair_v47.exe hit Windows STATUS_STACK_OVERFLOW (0xC00000FD). Check the last Startup: phase above; this build keeps HairApp off the main-thread stack.'
        }
        throw ('vulkan_hair_v47.exe exited with code ' + $exitCode)
    }
    $exitCode = 0
}
catch {
    Write-Host ''
    Write-Host ('[ERROR] ' + $_.Exception.Message) -ForegroundColor Red
    Write-Host ('Full transcript: ' + $log) -ForegroundColor Yellow
    if (Test-Path $fatal) {
        Write-Host ('Runtime fatal:   ' + $fatal) -ForegroundColor Yellow
    }
    $exitCode = 1
}
finally {
    try { Stop-Transcript | Out-Null } catch {}
}
exit $exitCode
