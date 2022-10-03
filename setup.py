from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="diskeyword",
    description="diskeyword package.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.0.1",
    packages=find_packages(),
    license="MIT",
)
