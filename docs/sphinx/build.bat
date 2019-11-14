
:: helper script for building locally, written by KSA
:: not sure if this is redundant with the makefile.
:: the makefile might only build the html, it might not run sphinx-apidoc

:: pip install sphinx_rtd_theme
:: pip install m2r
:: pip install nbsphinx
:: pip install nbsphinx-link

rmdir /s /q .\docs\sphinx\build
mkdir .\docs\sphinx\build

:: sphinx-apidoc -f -o docs/sphinx/source ./rdtools /separate

sphinx-build -b html docs/sphinx/source docs/sphinx/build

xcopy /I .\screenshots .\docs\sphinx\build\screenshots