from setuptools import setup

with open("README.md", "r") as fp:
    LONG_DESCRIPTION = fp.read()

REQUIREMENTS = ["numpy", "matplotlib", "scikit-learn", "mlxtend", "pandas", "joblib"]

setup(
    name="pyROGER",
    version="2.0.1",
    description="Implementation of ROGER in Python",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Martin de los Rios",
    author_email=" martindelosrios13@gmail.com ",
    url=" https://github.com/martindelosrios/pyROGER",
    py_modules=["ez_setup"],  # < - - - - - - - aca van los modulos
    packages=["pyROGER", "pyROGER/dataset"],  # < - -- - - - - aca van los paquetes
    exclude_package_data={"": ["tests"]},
    include_package_data=True,  # < - - - - - -- solo si hay datos
    license="The MIT License",
    install_requires=REQUIREMENTS,
    keywords=["pyROGER", "backsplash", "galaxies"],
    classifiers=[
        " Development Status :: 4 - Beta",
        " Intended Audience :: Education",
        " Intended Audience :: Science/Research",
        " License :: OSI Approved :: MIT License",
        " Operating System :: OS Independent",
        " Programming Language :: Python",
        " Programming Language :: Python :: 3.8",
        " Programming Language :: Python :: 3.9",
        " Programming Language :: Python :: 3.10",
        " Programming Language :: Python :: 3.11",
        " Programming Language :: Python :: 3.12",
        " Programming Language :: Python :: Implementation :: CPython",
        " Topic :: Scientific/Engineering",
    ],
)
