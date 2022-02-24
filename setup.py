from setuptools import setup, find_packages

setup(
    name="pvrpm",
    description="Run simulations to estimate LCOE costs for PV systems using SAM.",
    version="1.7.2",
    packages=find_packages(include=["pvrpm", "pvrpm.*"]),
    install_requires=[
        "nrel-pysam==3.0.0",
        "pandas",
        "click",
        "pyyaml",
        "scipy",
        "tqdm",
        "matplotlib",
        "python-dateutil",
    ],
    extras_require={
        "docs": [
            "sphinx",
            "sphinx_rtd_theme",
            "sphinxcontrib-napoleon",
        ],
        "testing": [
            "pytest",
        ],
    },
    python_requires=">=3.8",
    entry_points="""
        [console_scripts]
        pvrpm=pvrpm.__main__:main
    """,
    author=["Brandon Silva", "Paul Lunis"],
    long_description="README.md",
    long_description_content_type="text/markdown",
    url="https://github.com/FSEC-Photovoltaics/pvrpm-lcoe",
)
