from cx_Freeze import setup, Executable
 
base = None    
executables = [Executable("hackcovy.py", base=base)]
packages = ["idna"]
options = {
    "build_exe": {    
        "packages": packages,
    },    
}
setup(
    name = "Hackcovy",
    options = options,
    version = "1",
    description = "Covid-19 and pneumonia diagnosis tool via x-ray image",
    executables = executables
)
