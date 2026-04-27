@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

REM ============================================================================
REM Upload CellViT to GitHub - clean slate version
REM Repo: https://github.com/lyujiandao/Ziji_Lyu_20513276_fyp.git
REM ============================================================================

set "REPO_URL=https://github.com/lyujiandao/Ziji_Lyu_20513276_fyp.git"
set "USER_NAME=lyujiandao"
set "USER_EMAIL=zijilyu0628@gmail.com"
set "BRANCH=main"

echo.
echo ============================================================
echo  Uploading to: %REPO_URL%
echo  Working dir : %CD%
echo ============================================================
echo.
pause

echo.
echo === Step 1: Check git installation ===
where git >nul 2>nul
if errorlevel 1 (
    echo ERROR: git is not installed or not in PATH.
    echo Please install git from https://git-scm.com/download/win and re-run.
    pause
    exit /b 1
)
git --version

echo.
echo === Step 2: Remove old git association ===
if exist ".git" (
    rmdir /s /q ".git"
    if exist ".git" (
        echo ERROR: failed to remove .git folder. Close any program using it and retry.
        pause
        exit /b 1
    )
    echo Removed existing .git folder.
) else (
    echo No existing .git folder. Skipping.
)

echo.
echo === Step 3: Tune git for large pushes ===
git config --global http.postBuffer 524288000
git config --global http.lowSpeedLimit 0
git config --global http.lowSpeedTime 999999
echo Buffer set to 500MB.

echo.
echo === Step 4: Initialize new repository ===
git init
if errorlevel 1 goto :error
git branch -M %BRANCH%
git config user.name  "%USER_NAME%"
git config user.email "%USER_EMAIL%"
echo Identity: %USER_NAME% ^<%USER_EMAIL%^>

echo.
echo === Step 5: Stage all files (this may take a moment) ===
git add -A
if errorlevel 1 goto :error

echo.
echo === Step 6: Check for files larger than 100MB ===
set "OVERSIZE_FOUND=0"
for /f "usebackq delims=" %%F in (`git ls-files`) do (
    if exist "%%F" (
        for %%S in ("%%F") do (
            set /a "SIZE_MB=%%~zS / 1048576"
            if !SIZE_MB! GEQ 100 (
                echo   [TOO BIG] !SIZE_MB! MB  --  %%F
                set "OVERSIZE_FOUND=1"
            )
        )
    )
)
if "!OVERSIZE_FOUND!"=="1" (
    echo.
    echo ERROR: Files above exceed GitHub's 100MB hard limit.
    echo Remove them or add to .gitignore, then re-run this script.
    pause
    exit /b 1
)
echo No files exceed 100MB. OK.

echo.
echo === Step 7: Commit ===
git commit -m "Initial commit: Dino-Nuclei (CellViT-based) implementation"
if errorlevel 1 goto :error

echo.
echo === Step 8: Add remote and push ===
git remote add origin %REPO_URL%
echo Pushing to %REPO_URL% ...
echo If a login window appears, sign in with your GitHub account.
echo This may take several minutes for a 300MB repository.
echo.
git push -u origin %BRANCH% --force
if errorlevel 1 goto :pusherror

echo.
echo ============================================================
echo  SUCCESS
echo  Repository: %REPO_URL%
echo ============================================================
echo.
pause
exit /b 0

:error
echo.
echo ERROR: a git command failed. See messages above.
pause
exit /b 1

:pusherror
echo.
echo ERROR: git push failed.
echo Common causes:
echo   - Authentication cancelled or failed
echo   - Network interrupted (try again, the script will resume from scratch safely)
echo   - A file exceeds 100MB (check above for warnings)
pause
exit /b 1