from setuptools import setup


def read_requirements(filename: str):
    with open(filename) as requirements_file:
        import re

        def fix_url_dependencies(req: str) -> str:
            """Pip and setuptools disagree about how URL dependencies should be handled."""
            m = re.match(
                r"^(git\+)?(https|ssh)://(git@)?github\.com/([\w-]+)/(?P<name>[\w-]+)\.git(@([\w-]+))?(#egg=(?P<package_name>[\w-]+))?",
                req
            )
            if m is None:
                return req
            else:
                package_name = m.group('package_name') or m.group('name')
                return f"{package_name} @ {req}"

        requirements = []
        for line in requirements_file:
            line = line.strip()
            if line.startswith("#") or len(line) <= 0:
                continue
            requirements.append(fix_url_dependencies(line))

    return requirements


setup(
    name="viper",
    description="Implementation of ViperGPT",
    packages=['viper'],
    package_data={'viper': ['data/**/*.txt', 'data/**/*.json']},
    install_requires=read_requirements("requirements.txt"),
    python_requires=">=3.7",
)
