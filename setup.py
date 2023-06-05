from setuptools import setup, find_packages


def read_requirements(filename: str):
    with open(filename) as requirements_file:
        requirements = []
        for line in requirements_file:
            line = line.strip()
            if line.startswith("#") or len(line) <= 0:
                continue
            requirements.append(line)

    return requirements


setup(
    name="viper",
    description="Implementation of ViperGPT",
    packages=find_packages(where="viper"),
    package_data={'viper': ['data/**/*.txt', 'data/**/*.json']},
    install_requires=read_requirements("requirements.txt"),
    python_requires=">=3.7",
)
