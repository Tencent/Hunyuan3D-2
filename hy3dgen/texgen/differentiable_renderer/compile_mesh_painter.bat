FOR /F "tokens=*" %%i IN ('python -m pybind11 --includes') DO SET PYINCLUDES=%%i
echo %PYINCLUDES%
g++ -O3 -Wall -shared -std=c++11 -fPIC %PYINCLUDES% mesh_processor.cpp -o mesh_processor.pyd -lpython3.12