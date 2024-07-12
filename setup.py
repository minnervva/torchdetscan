from setuptools import setup

exec(open('src/minnervva/__version__.py').read())

# Use the README as the long_description
with open("README.md", "r") as f:
    long_description = f.read()

setup(name='MINNERVVA',
      version=__version__,
      packages=[''],
      python_requires=">=3.7.0",
      scripts=['src/minnervva/minnervva.py'],
      url='https://github.com/minnervva/minnervva',
      long_description=long_description,
      long_description_content_type='text/markdown',
      license='MIT License',
      author='Ada Sedova, Mark Coletti, Wael Elwasif',
      author_email='sedovaaa@ornl.gov, colettima@ornl.gov, elwasifwr@ornl.gov',
      description='Finding non-deterministic functions in pytorch code',
      entry_points={'console_scripts': ['minnervva = minnervva:main']},
      install_requires=[
            'rich',
      ]
      )
