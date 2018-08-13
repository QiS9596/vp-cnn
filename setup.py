from setuptools import setup, find_packages
# To use a consistent encoding
# from codecs import open
# from os import path

# here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
# with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
#     long_description = f.read()

setup(
    name='vpcnn',

    version='0.0.1',
    description='Text CNNs for Virtual Patient question classification',
    # long_description=long_description,

    # The project's main homepage.
    url='https://github.com/jaffe59/vp-cnn',

    # Author details
    author='Lifeng Jin, Evan Jaffe',
    
    # Choose your license
    license='Apache 2.0',

    packages=find_packages(),
    
    python_requires='>=3',
    install_requires=['torch', 'torchtext', 'numpy', 'tqdm'],

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.

    package_data={
        'vpcnn': ['data/*'],
    },

)
