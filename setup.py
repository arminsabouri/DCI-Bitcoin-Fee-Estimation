from setuptools import setup, find_packages

setup(
    name="data-lake",
    version="1.0",
    description="Data Lake",
    author="Armin Sabouri",
    author_email="me@arminsabouri.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    entry_points={
        "console_scripts": ["data-lake=data_lake.main:_entrypoint"],
    },
)
