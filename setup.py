from setuptools import setup

# TODO We need to decide if each tool has its own version or the whole package
# does.  I'm leaning towards the latter.
exec(open('src/torchdetscan/__version__.py').read())

# Use the README as the long_description
with open("README.md", "r") as f:
    long_description = f.read()

setup(name='torchdetscan',
      version=__version__,
      packages=['torchdetscan', 'torchdettest'],
      package_dir={'torchdetscan': 'src/torchdetscan',
                   'torchdettest': 'src/torchdettest'},
      py_modules=['torchdetscan.torchdetscan',
                  'torchdettest.torchdettest'],
      scripts=['src/torchdetscan/torchdetscan.py',
               'src/torchdettest/torchdettest.py'],
      python_requires=">=3.7.0",
      url='https://github.com/minnervva/torchdetscan',
      long_description=long_description,
      long_description_content_type='text/markdown',
      license='MIT License',
      author='Ada Sedova, Mark Coletti, Wael Elwasif',
      author_email='sedovaaa@ornl.gov, colettima@ornl.gov, elwasifwr@ornl.gov',
      description='Finding and testing for non-deterministic functions in pytorch code',
      entry_points={'console_scripts': ['torchdetscan = torchdetscan.torchdetscan:main',
                                        'torchdettest = torchdettest.torchdettest:main']},
      install_requires=[
            'rich',
      ],
      classifiers=["Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent", ],
      )
