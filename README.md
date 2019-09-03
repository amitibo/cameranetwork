CameraNetwork
=============

Code for running and analyzing the Camera Network

~~Latest version can be downloaded from [bitbucket](http://bitbucket.org/amitibo/CameraNetwork_git).~~

Documentation
-------------

Documentation is provided using [sphinx](http://www.sphinx-doc.org/).
To compile the documentation:

Navigate to document folder `cd docs` then
Generate source files. \n
  Windows:
 ```sh
 sphinx-apidoc -f -o source ..\CameraNetwork
 ```
  Linux:
  ```sh
 sphinx-apidoc -f -o source/ ../CameraNetwork
 ```
 Create html/pdf document
  For html version:
 ```sh
 make html
 ```
  For PDF:
  First run:
  ```sh
  sudo apt-get install texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra latexmk texlive-luatex texlive-xetex
  ```
  then
```sh
 make latexpdf
```

Author
------

Amit Aides

License
-------

Please see the LICENSE file for details on copying and usage.
