pip3 install twine wheel
copy ..\..\LICENSE.txt package
copy ..\..\build\install\Python_Examples\MalmoPython.lib package\malmo
copy ..\..\build\install\Python_Examples\MalmoPython.pyd package\malmo
copy ..\..\build\install\Python_Examples\malmoutils.py package\malmo
copy ..\..\build\install\Python_Examples\run_mission.py package\malmo
cd package
python setup.py bdist_wheel
del package\malmo\MalmoPython.lib package\malmo\MalmoPython.pyd
twine upload package/dist/*