from setuptools import setup, find_packages
from subprocess import Popen, PIPE

version = ''
sha = None
git_describe_process = Popen(
    ("git",
     "describe",
     "--tags"),
    stdout=PIPE,
    stderr=PIPE)
try:
    out, _ = git_describe_process.communicate()
    version = out.decode("utf-8")
    sp = version.split("-")
    version = sp[0]
    if version == '':
        version = '1.1'
    # Clean tag?
    if len(sp) != 0:
        commits = sp[1]
        sha = sp[2]
        version += "."+commits
    else:
        sha = None
except Exception:
    pass

# Provide better description
description="Trata"
if sha is not None:
    description += " (sha: {})".format(sha)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name="trata",
      version=version,
      description=description,
      url="https://github.com/LLNL/trata",
      author="Renee Olson",
      author_email="olson59@llnl.gov",
      long_description=long_description,
      long_description_content_type="text/markdown",
      license="BSD 3-Clause",
      packages=find_packages(),
      zip_safe=False,
      install_requires=[
            "numpy",
            "scikit-learn",
            "scipy",
            "matplotlib",
            "kosh",
            "h5py",
      ],
      classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
      ],
      python_requires=">=3.6",
      )
