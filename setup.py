from setuptools import setup

exec(open('src/torchdetscan/__version__.py').read())

# Use the README as the long_description
with open("README.md", "r") as f:
    long_description = f.read()

setup(name='torchdetscan',
      version=__version__,
      packages=[''],
      python_requires=">=3.7.0",
      scripts=['src/torchdetscan/torchdetscan.py'],
      url='https://github.com/minnervva/torchdetscan',
      long_description=long_description,
      long_description_content_type='text/markdown',
      license='MIT License',
      author='Ada Sedova, Mark Coletti, Wael Elwasif',
      author_email='sedovaaa@ornl.gov, colettima@ornl.gov, elwasifwr@ornl.gov',
      description='Finding non-deterministic functions in pytorch code',
      entry_points={'console_scripts': ['torchdetscan = torchdetscan:main']},
      install_requires=[
            'rich',
      ]
      )
