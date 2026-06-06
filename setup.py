from setuptools import setup, find_packages

setup(
    name="ifc-graphrag-dt",
    version="0.1.0",
    author="Prashant Srivastava",
    author_email="prashant_25s21res85@iitp.ac.in",
    description=(
        "IFC-GraphRAG-DT: Ontology-Grounded 3D Asset Generation "
        "for Building Digital Twins"
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/prashantsrivastava/ifc-graphrag-dt",
    packages=find_packages(exclude=["tests*", "notebooks*", "outputs*"]),
    python_requires=">=3.10",
    install_requires=open("requirements.txt").read().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    entry_points={
        "console_scripts": [
            "graphrag-dt=pipeline.run_pipeline:main",
            "dtah-eval=evaluation.run_eval:main",
        ]
    },
)
