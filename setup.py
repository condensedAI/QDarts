import setuptools

#EDIT THIS FILE!

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="opyrators",
    version="1.0.0",
    author="Evert van Nieuwenburg",
    author_email="evert.v.nieuwenburg@gmail.com",
    description="Manipulate quantum many-body operators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/everthemore/opyrators",
    packages=setuptools.find_packages(),
    setup_requires=["numpy"],
    install_requires=["numpy"],
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=2.7',
)