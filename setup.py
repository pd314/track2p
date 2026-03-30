from setuptools import find_packages, setup

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Core dependencies
core_requires = [
    'numpy>=2.0.2',
    'matplotlib>=3.9.4',
    'scikit-image>=0.24.0',
    'itk>=5.4.3',
    'tqdm>=4.67.1',
    'scikit-learn>=1.6.1',
    'openTSNE>=1.0.2',
    'pandas>=2.2.3',
    'itk-elastix>=0.23.0'
]

# GUI options
gui_requires = [
    'qtpy>=2.4.3',
    'PyQt6'
]

setup(
    name='track2p',
    version='0.6.2',
    packages=find_packages(),
    install_requires=core_requires,
    extras_require={
        'gui': gui_requires,
        'all': core_requires + gui_requires  # includes everything
    },
    long_description=long_description,
    long_description_content_type='text/markdown',
    include_package_data=True,
    package_data={
        '': ['resources/logo.png'],
    },
)