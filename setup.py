from setuptools import setup, find_packages

VERSION = '0.0.2' 
DESCRIPTION = 'FocalNet implementation in TensorFlow'
# LONG_DESCRIPTION = 'Implem'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="focalnet_tensorflow", 
        version=VERSION,
        author="Anas Raza",
        author_email="memanasraza@gmail.com",
        description=DESCRIPTION,
        # long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['tensorflow', 'tensorflow-addons'], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['focalnet', 'tensorflow'],
        
)