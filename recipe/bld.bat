
ROBOCOPY /E "%RECIPE_DIR%\.." "%SRC_DIR%" /XD .git /XD Wrappers\Python\build

mkdir "%SRC_DIR%\build_framework"

cd "%SRC_DIR%\build_framework"
cmake -G "NMake Makefiles" %RECIPE_DIR%\..\ -DCONDA_BUILD=ON -DCMAKE_BUILD_TYPE="Release" -DLIBRARY_LIB=%CONDA_PREFIX%\lib -DLIBRARY_INC=%CONDA_PREFIX% -DCMAKE_INSTALL_PREFIX=%PREFIX%

cmake --build . --target install
if errorlevel 1 exit 1
