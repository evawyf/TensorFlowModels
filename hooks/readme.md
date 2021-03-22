# About 
This folder contains a pre-commit hook that runs the following 
modules in bash when a user types "git commit":
- isort
- mypy
- pylint.sh 

# Dependencies 
You will need to have the following installed on your machine:
- isort (use "pip install")
- mypy (use "pip install")
- pylint (use "pip install")

# Setup
You will need to have the following in the root directory of your project:
- ```hooks``` directory, which contains the ```pre-commit``` hook script
- ```pylint.sh``` script
- ```pylintrc``` config file 

In order to get the pre-commit hook to work on your local machine, you can either
- copy and paste the ```pre-commit``` script into your local ```.git/hooks``` directory,
OR 
- create a symbolic link from the ```hooks``` directory to the ```.git/hooks``` directory
using the following command:
```ln -s -f ../../hooks/pre-commit .git/hooks/pre-commit```

# Important Notes
- You need to make sure the ```pre-commit``` and the ```pylint.sh``` files are 
executables; if they're not, use the command ```sudo chmod +x {FILENAME}``` to
make them executables
- If you run into permission issues with "git add . " command when running the pre-commit hook, 
use the following command to fix the issue: ```sudo chown -R "${USER:-$(id -un)}" . ```