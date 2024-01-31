from setuptools import setup, find_packages


def read_requirements():
    with open("requirements.txt", "r") as req:
        content = req.read()
        requirements = content.split("\n")

    return requirements


setup(
    name="operant-operation",
    version="0.1",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=read_requirements(),
    author="Ruairi O'Sullivan",
    author_email="ruairi.osullivan.work@gmail.com",
    description="Analysis for a set of operant conditioning neuroscience experiments.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Ruairi-osul/operant-operation",
)
